"""
Runs metamer generation on all the layers specified in build_network.py
Removes additional layers from saving to reduce the file size

You can either make a copy of this file into the directory with the build_network.py file
and run it from there directly, or specify the model directory containing the build_network.py
file as an argument (-D). The script will create a folder structure for the generated metamers
in the directory specified, or in the directory it is called from, if no directory is specified.

One of the most important files in the repo!
NOTE: Vision only!
"""

import os
import pickle
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models.feature_extraction as tv_fe
from PIL import Image

from analysis_scripts.input_helpers import generate_import_image_functions
from replicate.attacker import AttackerModel
from replicate.datasets import ImageNet


class SingleFeatureExtractor(torch.nn.Module):
    """
    Feature extractor that returns the activations of a single layer,
    so it can be used in those libraries for representation inversion
    (they only expect a single tensor as output of a forward call).

    Extractable layers can be found using `torchvision.models.feature_extraction.get_graph_node_names()`.
    """

    def __init__(self, model: torch.nn.Module, layer: str):
        super().__init__()
        self.model = tv_fe.create_feature_extractor(model, [layer])
        self.layer = layer

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)[self.layer]


def rescale_image(image, image_dict):
    """The image into the pytorch model should be between 0-1"""
    if image_dict['max_value_image_set'] == 255:
        image = image / 255.
    return image


def calc_loss(model: AttackerModel, inp, target, custom_loss, should_preproc=True):
    """
    Modified from the Attacker module of Robustness.
    Calculates the loss of an input with respect to target labels
    Uses custom loss (if provided) otherwise the criterion
    """
    if should_preproc:
        inp = model.preproc(inp)  # AttackerModel.preproc
    return custom_loss(model.model, inp, target)


class InputNormalize(torch.nn.Module):
    def __init__(self, new_mean, new_std, min_value, max_value):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = torch.clamp(x, self.min_value, self.max_value)
        x_normalized = (x - self.new_mean) / self.new_std
        return x_normalized


def inversion_loss_feathers(model, inp, targ, normalize_loss=True):
    activations = model(inp)
    rep = activations.contiguous().view(activations.size(0), -1)
    if normalize_loss:
        loss = torch.div(
            torch.norm(rep - targ, dim=1),
            torch.norm(targ, dim=1)
        )
    else:
        loss = torch.norm(rep - targ, dim=1)
    return loss, None


def run_image_metamer_generation(image_id, seed, output_name: Optional[str] = None):
    iterations = 1000
    overwrite_pckl = True
    input_image_func_name = "400_16_class_imagenet_val"
    num_repetitions = 8
    model_directory = f"model_analysis_folders/visual_networks/alexnet"

    predictions_out_dict = {}
    rep_out_dict = {}
    all_outputs_out_dict = {}
    xadv_dict = {}
    all_losses_dict = {}
    predicted_labels_out_dict = {}
    predicted_16_cat_labels_out_dict = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preproc = InputNormalize(
        new_mean=torch.tensor([0.485, 0.456, 0.406], device=device),
        new_std=torch.tensor([0.229, 0.224, 0.225], device=device),
        min_value=0.0,
        max_value=1.0,
    )

    BATCH_SIZE = 1

    torch.manual_seed(seed)
    np.random.seed(seed)

    # init model (and feature extractor)
    layer_to_invert = "features.9"  # relu3
    class_model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
    fe_model = SingleFeatureExtractor(class_model, layer_to_invert)
    ds = ImageNet()
    model = AttackerModel(fe_model, ds, preproc=preproc)
    assert isinstance(model, torch.nn.Module), "model is no valid torch module"
    assert type(model) is AttackerModel, "model is no valid robustness attacker model"

    # Load dataset for metamer generation
    # this loads the image.
    input_image_func = generate_import_image_functions(input_image_func_name, data_format='NCHW')
    image_dict = input_image_func(image_id)  # load the image from the reduced dataset (400 images, see /assets)
    image_dict['image_orig'] = image_dict['image']
    # Preprocess to be in the format for pytorch
    image_dict['image'] = rescale_image(image_dict['image'], image_dict)  # essentially a ToTensor transform
    scale_image_save_PIL_factor = 255
    init_noise_mean = 0.5
    # Add a batch dimension to the input image
    # finally turn into a pytorch tensor
    reference_image = torch.tensor(np.expand_dims(image_dict['image'], 0)).float().contiguous()

    # Label name for the 16 way imagenet task
    label_name = image_dict['correct_response']

    # Set up the saving folder and make sure that the file doesn't already exist
    synth_name = input_image_func_name + '_RS%d' % seed + '_I%d' % iterations + '_N%d' % num_repetitions
    if output_name is None:
        base_filepath = os.path.join(model_directory, "metamers", synth_name, f"%{image_id}_SOUND_{label_name}") + "/"
    else:
        base_filepath = os.path.join(model_directory, "metamers", output_name) + "/"
    pckl_path = base_filepath + '/all_metamers_pickle.pckl'
    os.makedirs(base_filepath, exist_ok=True)

    if os.path.isfile(pckl_path) and not overwrite_pckl:
        raise FileExistsError('The file %s already exists, and you are not forcing overwriting' % pckl_path)

    # Send model to GPU (b/c we haven't loaded a model, so it is not on the GPU)
    model: AttackerModel = model.cuda()
    model.eval()
    class_model = class_model.cuda().eval()

    with torch.no_grad():
        reference_activations, reference_image = model(reference_image.cuda())  # attention: this does preprocessing
        reference_output = class_model(reference_image.cuda())

    # Calculate human-readable labels and 16 category labels for the original image
    reference_predictions = []
    for b in range(BATCH_SIZE):
        reference_predictions.append(reference_output[b].detach().cpu().numpy())

    # Get the predicted 16 category label
    pprint([x.shape for x in reference_predictions])

    # Make the noise input (will use for all the input seeds)
    # the noise scale is typically << the noise mean, so we don't have to worry about negative values.
    initial_noise = (torch.randn_like(reference_image) * 1 / 20 + 0.5)

    # Choose the inversion parameters (will run 4x the iterations, reducing the learning rate each time)
    # will be passed to Attacker to generate adv example
    synth_kwargs = {
        # same simple loss as in the paper
        'custom_loss': inversion_loss_feathers,
        'eps': 100000,  # why this high? this is weird, usually 8/255 or some is used
        'step_size': 1.0,
        # essentially works like learning rate. halved every 3000 iterations (default: 1.0)
        'iterations': iterations,  # iterations to generate one adv example
        'targeted': True,
    }

    metamer = torch.clamp(initial_noise, ds.min_value, ds.max_value).cuda()

    reference_representation = reference_activations.contiguous().view(reference_activations.size(0), -1)

    with open(os.path.join(os.path.dirname(pckl_path), "initial_values.pckl"), 'wb') as handle:
        data = {
            "metamer": metamer.detach().cpu(),
            "reference_image": reference_image.detach().cpu(),
            "reference_activations": reference_activations.detach().cpu(),
            "reference_representation": reference_representation.detach().cpu(),
            "reference_output": reference_output.detach().cpu(),
        }
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do the optimization, and save the losses occasionally
    all_losses = {}

    # Note for loss calculation: it takes in the target representation but only the prediction input
    # Internally it calculates the output and its according representation.
    # This causes an unnecessary forward pass of the entire model as we do another one below to optimize ...
    this_loss, _ = calc_loss(model, metamer, reference_representation.clone(), synth_kwargs['custom_loss'])
    all_losses[0] = this_loss.detach().cpu()

    print('Step %d | Layer %s | Loss %f' % (0, layer_to_invert, this_loss))

    prediction_activations, adv_ex = model(
        metamer,
        reference_representation.clone(),
        make_adv=True,
        **synth_kwargs,
    )
    with torch.no_grad():
        prediction_output = class_model(metamer.clone())

    this_loss, _ = calc_loss(model, adv_ex, reference_representation.clone(), synth_kwargs['custom_loss'])
    all_losses[synth_kwargs['iterations']] = this_loss.detach().cpu()

    print('Step %d | Layer %s | Loss %f' % (synth_kwargs['iterations'], layer_to_invert, this_loss))

    # this iteration is the interesting part!
    # it is quite simple and basically improves the adversarial example
    # multiple times using PGD attack until it becomes the metamer
    for i in range(num_repetitions - 1):
        metamer = adv_ex
        synth_kwargs['step_size'] = synth_kwargs['step_size'] / 2  # constrain max L2 norm of grad desc desc
        prediction_activations, adv_ex = model(
            metamer,
            reference_representation.clone(),
            make_adv=True,
            **synth_kwargs,
        )  # Image inversion using PGD
        prediction_output = class_model(adv_ex.clone())

        this_loss, _ = calc_loss(model, adv_ex, reference_representation.clone(),
                                 synth_kwargs['custom_loss'])
        all_losses[(i + 2) * synth_kwargs['iterations']] = this_loss.detach().cpu()

        print('Step %d | Layer %s | Loss %f' % (synth_kwargs['iterations'] * (i + 2), layer_to_invert, this_loss))

    print("all iteration steps completed!")
    print("losses:", all_losses)

    with open(os.path.join(os.path.dirname(pckl_path), "simple_output.pckl"), 'wb') as handle:
        data = {
            "x_adv": adv_ex.detach().cpu(),
            "reference_image": reference_image.detach().cpu(),
            "activations": prediction_activations.detach().cpu(),
            "reference_activations": reference_activations.detach().cpu(),
        }
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

    ########################################

    prediction_representation = prediction_activations.contiguous().view(prediction_activations.size(0), -1)

    # save last prediction for later evaluation
    predictions_out_dict[layer_to_invert] = prediction_output.detach().cpu()

    # save last representation for later evaluation
    try:  # should be None though
        rep_out_dict[layer_to_invert] = prediction_representation.detach().cpu()
    except AttributeError:
        rep_out_dict[layer_to_invert] = prediction_representation

    # clean up and save last prediction activations for later evaluation
    prediction_activations = prediction_activations.detach().cpu()

    # Calculate the predictions and save them in the dictionary
    synth_predictions = []
    for b in range(BATCH_SIZE):
        try:
            synth_predictions.append(prediction_output[b].detach().cpu().numpy())
        except KeyError:
            synth_predictions.append(reference_output['signal/word_int'][b].detach().cpu().numpy())

    # Get the predicted 16 category label

    all_outputs_out_dict[layer_to_invert] = prediction_activations  # includes all activations, not output!
    xadv_dict[layer_to_invert] = adv_ex.detach().cpu()
    all_losses_dict[layer_to_invert] = all_losses
    predicted_labels_out_dict[layer_to_invert] = synth_predictions

    ###############################
    # COMPUTATION DONE ############
    ###############################
    print("iteration/computation complete, start evaluation")

    # add params and outputs to pckl
    pckl_output_dict = {}
    # outputs
    pckl_output_dict['xadv_dict'] = xadv_dict
    pckl_output_dict['all_losses'] = all_losses  # last_loss
    pckl_output_dict['all_outputs_out_dict'] = all_outputs_out_dict
    pckl_output_dict['predicted_labels_out_dict'] = predicted_labels_out_dict
    pckl_output_dict['predicted_16_cat_labels_out_dict'] = predicted_16_cat_labels_out_dict
    pckl_output_dict['predictions_out_dict'] = predictions_out_dict
    pckl_output_dict['rep_out_dict'] = rep_out_dict
    pckl_output_dict['orig_predictions'] = reference_predictions
    # params
    pckl_output_dict['image_dict'] = image_dict  # reference image
    pckl_output_dict['metamer_layers'] = [layer_to_invert]
    pckl_output_dict['RANDOMSEED'] = seed
    pckl_output_dict['ITERATIONS'] = iterations
    pckl_output_dict['NUMREPITER'] = num_repetitions
    pckl_output_dict['NOISE_SCALE'] = noise_scale
    pckl_output_dict['step_size'] = initial_step_size

    # clean up and save reference activations for later evaluation
    reference_activations = reference_activations.detach().cpu()
    pckl_output_dict['all_outputs_orig'] = reference_activations

    # clean up and save reference output for later evaluation
    reference_output = reference_output.detach().cpu()
    pckl_output_dict['predictions_orig'] = reference_output

    if reference_representation is not None:
        reference_representation = reference_representation.detach().cpu()
    pckl_output_dict['rep_orig'] = reference_representation
    pckl_output_dict['sound_orig'] = reference_image.detach().cpu()

    # Just use the name of the loss to save synth_kwargs don't save the function
    pckl_output_dict['synth_kwargs'] = synth_kwargs

    with open(pckl_path, 'wb') as handle:
        pickle.dump(pckl_output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###############################
    # RESULT COLLECTION DONE ######
    ###############################
    print("result collection done", list(pckl_output_dict.keys()))
    print("Start making plots ...")

    # Make plots for each layer and save the image files
    layer_filepath = base_filepath + f"layer_{layer_to_invert}"

    # collect all relevant variables
    prediction_output = predictions_out_dict[layer_to_invert]
    adv_ex = xadv_dict[layer_to_invert]
    prediction_activations = all_outputs_out_dict[layer_to_invert]

    fig = plt.figure(figsize=(BATCH_SIZE * 5, 12))
    for i in range(BATCH_SIZE):  # should only be one iteration lol
        # Get labels to use for the plots
        try:
            reference_predictions = reference_output[i].detach().cpu().numpy()
            synth_predictions = prediction_output[i].detach().cpu().numpy()
        except KeyError:
            reference_predictions = reference_output['signal/word_int'][i].detach().cpu().numpy()
            synth_predictions = reference_output['signal/word_int'][i].detach().cpu().numpy()

        orig_label = np.argmax(reference_predictions)
        synth_label = np.argmax(synth_predictions)

        # results collected, start plotting!

        # plot original image
        plt.subplot(3, BATCH_SIZE, i + 1)  # nrows, ncols, index
        if reference_image[i].shape[0] == 3:
            plt.imshow((np.rollaxis(np.array(reference_image[i].cpu().numpy()), 0, 3)), interpolation='none')
        elif reference_image[i].shape[0] == 1:
            plt.imshow((np.array(reference_image[i].cpu().numpy())[0, :, :]), interpolation='none', cmap='gray')
        else:
            raise ValueError('Image dimensions are not appropriate for saving. Check dimensions')
        plt.title('Layer %s, Image %d \n Orig Coch' % (layer_to_invert, i,))

        # plot adversarial example
        plt.subplot(3, BATCH_SIZE, BATCH_SIZE + i + 1)  # nrows, ncols, index
        if reference_image[i].shape[0] == 3:
            plt.imshow((np.rollaxis(np.array(adv_ex[i].cpu().numpy()), 0, 3)), interpolation='none')
        elif reference_image[i].shape[0] == 1:
            plt.imshow((np.array(adv_ex[i].cpu().numpy())[0, :, :]), interpolation='none', cmap='gray')
        else:
            raise ValueError('Image dimensions are not appropriate for saving. Check dimensions')
        plt.title('Layer %s, Image %d" \n Synth Coch' % (layer_to_invert, i))

        # scatter plot for current layer (current orig vs current synth)
        plt.subplot(3, BATCH_SIZE, BATCH_SIZE * 2 + i + 1)  # nrows, ncols, index
        plt.scatter(
            np.ravel(np.array(reference_activations.cpu())[i, :]),
            np.ravel(prediction_activations.cpu().detach().numpy()[i, :])
        )
        plt.title('Layer %s, Image %d \n Optimization' % (layer_to_invert, i))
        plt.xlabel('Orig Activations (%s)' % layer_to_invert)
        plt.ylabel('Synth Activations (%s)' % layer_to_invert)

        fig.savefig(layer_filepath + '_image_optimization.png')

        plt.close()

        # Only save the individual generated image if the layer optimization succeeded
        if override_save:
            if reference_image[i].shape[0] == 3:
                synth_image = Image.fromarray(
                    (np.rollaxis(np.array(adv_ex[i].cpu().numpy()), 0, 3) * scale_image_save_PIL_factor).astype(
                        'uint8'))
            elif reference_image[i].shape[0] == 1:
                synth_image = Image.fromarray(
                    (np.array(adv_ex[i].cpu().numpy())[0] * scale_image_save_PIL_factor).astype('uint8'))
            synth_image.save(layer_filepath + '_synth.png', 'PNG')
