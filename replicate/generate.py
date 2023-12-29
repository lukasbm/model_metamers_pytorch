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

import csv
import os
import pickle
from dataclasses import dataclass
from pprint import pprint
from typing import Optional, Literal

import simple_parsing
import torch
import torchvision
from PIL import Image
from matplotlib import pylab as plt

from analysis_scripts.default_paths import WORDNET_ID_TO_HUMAN_PATH, IMAGENET_PATH
from analysis_scripts.helpers_16_choice import force_16_choice
from analysis_scripts.input_helpers import generate_import_image_functions
from robustness import datasets
from robustness.tools.distance_measures import *
from robustness.tools.label_maps import CLASS_DICT
from .attacker import AttackerModel
from .custom_synthesis_losses import InversionLossLayerReplica


class SingleFeatureExtractor(torch.nn.Module):
    """
    Feature extractor that returns the activations of a single layer,
    so it can be used in those libraries for representation inversion
    (they only expect a single tensor as output of a forward call).

    Extractable layers can be found using `torchvision.models.feature_extraction.get_graph_node_names()`.
    """

    def __init__(self, model: torch.nn.Module, layer: str):
        super().__init__()
        self.model = torchvision.models.feature_extraction.create_feature_extractor(model, [layer])
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


def run_image_metamer_generation(image_id, loss_func_name, input_image_func_name, seed, overwrite_pckl,
                                 use_dataset_preproc, initial_step_size, noise_scale, iterations, num_repetitions,
                                 override_save, model_directory,
                                 initial_metamer: Literal["noise", "reference", "uniform", "constant"] = "noise",
                                 output_name: Optional[str] = None, fake_relu: bool = True):
    """
    @param fake_relu: disable the usage of fake relu.
        This would otherwise use simplified gradients to improve optimization.
        NOTE: remember to exclude fake_relu layers from the models metamer_layers in the respective build_network.py
    """
    predictions_out_dict = {}
    rep_out_dict = {}
    all_outputs_out_dict = {}
    xadv_dict = {}
    all_losses_dict = {}
    predicted_labels_out_dict = {}
    predicted_16_cat_labels_out_dict = {}

    BATCH_SIZE = 1

    torch.manual_seed(seed)
    np.random.seed(seed)

    # init model (and feature extractor)
    layer_to_invert = "features.9"
    class_model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
    fe_model = SingleFeatureExtractor(class_model, layer_to_invert)
    ds = datasets.ImageNet(IMAGENET_PATH)
    model = AttackerModel(fe_model, ds)
    assert isinstance(model, torch.nn.Module), "model is no valid torch module"
    assert type(model) is AttackerModel, "model is no valid robustness attacker model"

    # Get the WNID
    with open(WORDNET_ID_TO_HUMAN_PATH, mode='r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        wnid_imagenet_name = {rows[0]: rows[1] for rows in reader}

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

    # Get the full imagenet keys for printing predictions
    label_keys = CLASS_DICT['ImageNet'].keys()
    label_values = CLASS_DICT['ImageNet'].values()
    label_idx = list(label_keys)[list(label_values).index(wnid_imagenet_name[image_dict['imagenet_category']])]
    targ = torch.from_numpy(np.array([label_idx])).float()

    # Set up the saving folder and make sure that the file doesn't already exist
    synth_name = input_image_func_name + '_' + loss_func_name + '_RS%d' % seed + '_I%d' % iterations + '_N%d' % num_repetitions
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
    reference_16_cat_prediction = [force_16_choice(np.flip(np.argsort(p.ravel(), 0)), CLASS_DICT['ImageNet'])
                                   for p in reference_predictions]  # p should be an array of shape [1000]
    print('Orig Image 16 Category Prediction: %s' % reference_16_cat_prediction)

    # Make the noise input (will use for all the input seeds)
    # the noise scale is typically << the noise mean, so we don't have to worry about negative values.
    initial_noise = (torch.randn_like(reference_image) * noise_scale + init_noise_mean)

    # Choose the inversion parameters (will run 4x the iterations, reducing the learning rate each time)
    # will be passed to Attacker to generate adv example
    synth_kwargs = {
        # same simple loss as in the paper
        'custom_loss': InversionLossLayerReplica(normalize_loss=True),
        'eps': 100000,  # why this high? this is weird, usually 8/255 or some is used
        'step_size': initial_step_size,
        # essentially works like learning rate. halved every 3000 iterations (default: 1.0)
        'iterations': iterations,  # iterations to generate one adv example
        'targeted': True,
    }

    # set up initial metamer
    if initial_metamer == "reference":
        metamer = reference_image.clone().cuda()
    elif initial_metamer == "noise":
        # Use same noise for every layer.
        metamer = torch.clamp(initial_noise, ds.min_value, ds.max_value).cuda()
    elif initial_metamer == "uniform":
        metamer = torch.distributions.Uniform(0, 1).sample(reference_image.shape).cuda()
    elif initial_metamer == "constant":
        metamer = torch.ones_like(reference_image) * 0.5
    else:
        raise ValueError("invalid initial metamer param")

    reference_representation = reference_activations.contiguous().view(reference_activations.size(0), -1)

    with open(os.path.join(os.path.dirname(pckl_path), "initial_values.pckl"), 'wb') as handle:
        data = {
            "metamer": metamer.detach().cpu(),
            "reference_image": reference_image.detach().cpu(),
            "reference_activations": reference_activations.detach().cpu(),
            "reference_representation": reference_representation.detach().cpu(),
            "reference_output": reference_output.detach().cpu(),
            "targ": targ.detach().cpu(),
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
    synth_16_cat_prediction = [force_16_choice(np.flip(np.argsort(p.ravel(), 0)),
                                               CLASS_DICT['ImageNet']) for p in synth_predictions]
    print('Layer %s, Synth Image 16 Category Prediction: %s' % (
        layer_to_invert, synth_16_cat_prediction))

    all_outputs_out_dict[layer_to_invert] = prediction_activations  # includes all activations, not output!
    xadv_dict[layer_to_invert] = adv_ex.detach().cpu()
    all_losses_dict[layer_to_invert] = all_losses
    predicted_labels_out_dict[layer_to_invert] = synth_predictions
    predicted_16_cat_labels_out_dict[layer_to_invert] = synth_16_cat_prediction

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
    pckl_output_dict['orig_16_cat_prediction'] = reference_16_cat_prediction
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

    # Just use the name of the loss to save synthkwargs don't save the function
    synth_kwargs['custom_loss'] = loss_func_name
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

        # Get the predicted 16 category label (re-evaluating for some reason??)
        reference_16_cat_prediction = force_16_choice(np.flip(np.argsort(reference_predictions.ravel(), 0)),
                                                      CLASS_DICT['ImageNet'])
        synth_16_cat_prediction = force_16_choice(np.flip(np.argsort(synth_predictions.ravel(), 0)),
                                                  CLASS_DICT['ImageNet'])
        print('Layer %s, Image %d, Orig Image 16 Category Prediction: %s' % (
            layer_to_invert, i, reference_16_cat_prediction))
        print('Layer %s, Image %d, Synth Image 16 Category Prediction: %s' % (
            layer_to_invert, i, synth_16_cat_prediction))

        # Set synthesis success based on the 16 category labels -- we can later evaluate the 1000 category labels.
        synth_success = reference_16_cat_prediction == synth_16_cat_prediction

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
        plt.title('Layer %s, Image %d, Predicted Orig Label "%s" \n Orig Coch' % (
            layer_to_invert, i,  # CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())]))
            CLASS_DICT['ImageNet'][int(orig_label)]))

        # plot adversarial example
        plt.subplot(3, BATCH_SIZE, BATCH_SIZE + i + 1)  # nrows, ncols, index
        if reference_image[i].shape[0] == 3:
            plt.imshow((np.rollaxis(np.array(adv_ex[i].cpu().numpy()), 0, 3)), interpolation='none')
        elif reference_image[i].shape[0] == 1:
            plt.imshow((np.array(adv_ex[i].cpu().numpy())[0, :, :]), interpolation='none', cmap='gray')
        else:
            raise ValueError('Image dimensions are not appropriate for saving. Check dimensions')
        plt.title('Layer %s, Image %d, Predicted Synth Label "%s" \n Synth Coch' % (
            layer_to_invert, i,  # CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())]))
            CLASS_DICT['ImageNet'][int(synth_label)]))

        # scatter plot for current layer (current orig vs current synth)
        plt.subplot(3, BATCH_SIZE, BATCH_SIZE * 2 + i + 1)  # nrows, ncols, index
        plt.scatter(
            np.ravel(np.array(reference_activations.cpu())[i, :]),
            np.ravel(prediction_activations.cpu().detach().numpy()[i, :])
        )
        plt.title('Layer %s, Image %d, Label "%s" \n Optimization' % (
            layer_to_invert, i, CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())]))
        plt.xlabel('Orig Activations (%s)' % layer_to_invert)
        plt.ylabel('Synth Activations (%s)' % layer_to_invert)

        fig.savefig(layer_filepath + '_image_optimization.png')

        # print the information also in the console
        try:
            print('Layer %s, Image %d, Label "%s", Prediction Orig "%s", Prediction Synth "%s"' % (
                layer_to_invert, i,
                CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())],
                CLASS_DICT['ImageNet'][int(np.argmax(reference_output[i].detach().cpu().numpy()))],
                CLASS_DICT['ImageNet'][int(np.argmax(prediction_output[i].detach().cpu().numpy()))]))
        except KeyError:
            print('Layer %s, Image %d, Label "%s", Prediction Orig "%s", Prediction Synth "%s"' % (
                layer_to_invert, i,
                CLASS_DICT['ImageNet'][int(targ[i].cpu().numpy())],
                CLASS_DICT['ImageNet'][
                    int(np.argmax(reference_output['signal/word_int'][i].detach().cpu().numpy()))],
                CLASS_DICT['ImageNet'][
                    int(np.argmax(prediction_output['signal/word_int'][i].detach().cpu().numpy()))]))

        plt.close()

        # Only save the individual generated image if the layer optimization succeeded
        if synth_success or override_save:
            if reference_image[i].shape[0] == 3:
                synth_image = Image.fromarray(
                    (np.rollaxis(np.array(adv_ex[i].cpu().numpy()), 0, 3) * scale_image_save_PIL_factor).astype(
                        'uint8'))
            elif reference_image[i].shape[0] == 1:
                synth_image = Image.fromarray(
                    (np.array(adv_ex[i].cpu().numpy())[0] * scale_image_save_PIL_factor).astype('uint8'))
            synth_image.save(layer_filepath + '_synth.png', 'PNG')


@dataclass(slots=True)
class MetamerGeneratorArgs:
    model_directory: str = "model_analysis_folders/visual_networks/alexnet"  # The directory with the location of the `build_network.py` file. Folder structure for saving metamers will be created in this directory. If not specified, assume this script is located in the same directory as the build_network.py file.
    sidx: int = 0  # index into the sound list for the time_average sound, range 0 to 400
    loss_func_name: str = "inversion_loss_layer"  # loss function to use for the synthesis
    input_image_func_name: str = "400_16_class_imagenet_val"  # function to use for grabbing the input image sources
    random_seed: int = 0  # random seed to use for synthesis
    iterations: int = 3000  # number of iterations in robustness synthesis kwargs (when make_adv=True)
    num_repetitions: int = 8  # number of repetitions to run the robustness synthesis, each time decreasing the learning rate by half
    override_save: bool = False  # set to true to save, even if the optimization does not succeed
    overwrite_pckl: bool = True  # set to true to overwrite the saved pckl file, if false then exits out if the file already exists
    use_dataset_preproc: bool = False  # preprocess the dataset (by default just something like tv.ToTensor)
    noise_scale: float = 1 / 20  # multiply the noise by this value for the synthesis initialization (noise init)
    step_size: float = 1.0  # Initial step size for the metamer generation
    intial_metamer: Literal["noise", "reference", "uniform", "constant"] = "noise"  # initial metamer value


def main():
    args: MetamerGeneratorArgs = simple_parsing.parse(MetamerGeneratorArgs)
    pprint(args)

    run_image_metamer_generation(
        image_id=args.sidx,
        loss_func_name=args.loss_func_name,
        input_image_func_name=args.input_image_func_name,
        seed=args.random_seed,
        overwrite_pckl=args.overwrite_pckl,
        use_dataset_preproc=args.use_dataset_preproc,
        initial_step_size=args.step_size,
        noise_scale=args.noise_scale,
        iterations=args.iterations,
        num_repetitions=args.num_repetitions,
        override_save=args.override_save,
        model_directory=args.model_directory,
        initial_metamer=args.intial_metamer
    )


if __name__ == '__main__':
    main()
