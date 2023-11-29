"""
this is a non-functional copy of make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py intended
for easier studying of the code.

NOTE: vision only
"""
import argparse
import importlib.util
import os

import torch
from PIL import Image

from analysis_scripts.input_helpers import generate_import_image_functions
from robustness import custom_synthesis_losses
from robustness.tools.distance_measures import *


def preproc_image(image, image_dict):
    """The image into the pytorch model should be between 0-1"""
    if image_dict['max_value_image_set'] == 255:
        image = image / 255.
    return image


def calc_loss(model, inp, target, custom_loss, should_preproc=True):
    """
    Modified from the Attacker module of Robustness. 
    Calculates the loss of an input with respect to target labels
    Uses custom loss (if provided) otherwise the criterion
    """
    if should_preproc:
        inp = model.preproc(inp)
    return custom_loss(model.model, inp, target)


def run_image_metamer_generation(SIDX, LOSS_FUNCTION, INPUTIMAGEFUNCNAME, RANDOMSEED, overwrite_pckl,
                                 use_dataset_preproc, step_size, NOISE_SCALE, ITERATIONS, NUMREPITER,
                                 OVERRIDE_SAVE, MODEL_DIRECTORY):
    build_network_spec = importlib.util.spec_from_file_location("build_network",
                                                                os.path.join(MODEL_DIRECTORY, 'build_network.py'))
    build_network = importlib.util.module_from_spec(build_network_spec)
    build_network_spec.loader.exec_module(build_network)

    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)

    model, ds, metamer_layers = build_network.main(return_metamer_layers=True)

    # Load dataset for metamer generation
    INPUTIMAGEFUNC = generate_import_image_functions(INPUTIMAGEFUNCNAME, data_format='NCHW')
    image_dict = INPUTIMAGEFUNC(SIDX)
    image_dict['image_orig'] = image_dict['image']
    # Preprocess to be in the format for pytorch
    image_dict['image'] = preproc_image(image_dict['image'], image_dict)
    if use_dataset_preproc:  # Apply for some models, for instance if we have greyscale images or different sizes.
        image_dict['image'] = ds.transform_test(Image.fromarray(np.rollaxis(np.uint8(image_dict['image'] * 255), 0, 3)))
        init_noise_mean = ds.init_noise_mean
    else:
        init_noise_mean = 0.5

    # Add a batch dimension to the input image 
    im = torch.tensor(np.expand_dims(image_dict['image'], 0)).float().contiguous()

    # Send model to GPU (b/c we haven't loaded a model, so it is not on the GPU)
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        (predictions, rep, all_outputs), orig_im = model(im.cuda(), with_latent=True,
                                                         fake_relu=True)  # Corresponding representation

    # Make the noise input (will use for all the input seeds)
    # the noise scale is typically << the noise mean, so we don't have to worry about negative values.
    im_n_initialized = (torch.randn_like(im) * NOISE_SCALE + init_noise_mean).detach().cpu().numpy()

    for layer_to_invert in metamer_layers:
        loss = custom_synthesis_losses.LOSSES[LOSS_FUNCTION](layer_to_invert, normalize_loss=True)
        # Choose the inversion parameters (will run 4x the iterations, reducing the learning rate each time)
        synth_kwargs = {
            'custom_loss': loss,
            'constraint': '2',
            'eps': 100000,
            'step_size': step_size,
            'iterations': ITERATIONS,
            'do_tqdm': False,
            'targeted': True,
            'use_best': False
        }

        # Use same noise for every layer.
        im_n = torch.clamp(torch.from_numpy(im_n_initialized), ds.min_value, ds.max_value).cuda()
        invert_rep = all_outputs[layer_to_invert].contiguous().view(all_outputs[layer_to_invert].size(0), -1)

        # Do the optimization, and save the losses occasionally
        this_loss, _ = calc_loss(model, im_n, invert_rep.clone(), loss)
        print('Step %d | Layer %s | Loss %f' % (0, layer_to_invert, this_loss))

        (predictions_out, rep_out, all_outputs_out), xadv = model(im_n, invert_rep.clone(), make_adv=True,
                                                                  **synth_kwargs, with_latent=True, fake_relu=True)
        this_loss, _ = calc_loss(model, xadv, invert_rep.clone(), loss)
        print('Step %d | Layer %s | Loss %f' % (synth_kwargs['iterations'], layer_to_invert, this_loss))
        for i in range(NUMREPITER - 1):
            im_n = xadv
            synth_kwargs['step_size'] = synth_kwargs['step_size'] / 2
            (predictions_out, rep_out, all_outputs_out), xadv = model(im_n, invert_rep.clone(), make_adv=True,
                                                                      **synth_kwargs, with_latent=True,
                                                                      fake_relu=True)  # Image inversion using PGD
            this_loss, _ = calc_loss(model, xadv, invert_rep.clone(), loss)
            print('Step %d | Layer %s | Loss %f' % (synth_kwargs['iterations'] * (i + 2), layer_to_invert, this_loss))


def main(raw_args=None):
    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Input the sound indices and the layers to match')
    parser.add_argument('SIDX', metavar='I', type=int, help='index into the sound list for the time_average sound')
    parser.add_argument('-L', '--LOSSFUNCTION', metavar='--L', type=str, default='inversion_loss_layer',
                        help='loss function to use for the synthesis')
    parser.add_argument('-F', '--INPUTIMAGEFUNC', metavar='--A', type=str, default='400_16_class_imagenet_val',
                        help='function to use for grabbing the input image sources')
    parser.add_argument('-R', '--RANDOMSEED', metavar='--R', type=int, default=0,
                        help='random seed to use for synthesis')
    parser.add_argument('-I', '--ITERATIONS', metavar='--I', type=int, default=3000,
                        help='number of iterations in robustness synthesis kwargs')
    parser.add_argument('-N', '--NUMREPITER', metavar='--N', type=int, default=8,
                        help='number of repetitions to run the robustness synthesis, each time decreasing the learning rate by half')
    parser.add_argument('-S', '--OVERRIDE_SAVE', metavar='--S', type=bool, default=False,
                        help='set to true to save, even if the optimization does not succeed')
    parser.add_argument('-O', '--OVERWRITE_PICKLE', action='store_true',
                        help='set to true to overwrite the saved pckl file, if false then exits out if the file already exists')
    parser.add_argument('-DP', '--DATASET_PREPROC', action='store_true', dest='use_dataset_preproc')
    parser.add_argument('-E', '--NOISE_SCALE', metavar='--E', type=float, default=1 / 20,
                        help='multiply the noise by this value for the synthesis initialization')
    parser.add_argument('-Z', '--STEP_SIZE', metavar='--Z', type=float, default=1,
                        help='Initial step size for the metamer generation')
    parser.add_argument('-D', '--DIRECTORY', metavar='--D', type=str, default=None,
                        help='The directory with the location of the `build_network.py` file. Folder structure for saving metamers will be created in this directory. If not specified, assume this script is located in the same directory as the build_network.py file.')

    args = parser.parse_args(raw_args)
    SIDX = args.SIDX
    LOSS_FUNCTION = args.LOSSFUNCTION
    INPUTIMAGEFUNCNAME = args.INPUTIMAGEFUNC
    RANDOMSEED = args.RANDOMSEED
    overwrite_pckl = args.OVERWRITE_PICKLE
    use_dataset_preproc = args.use_dataset_preproc
    step_size = args.STEP_SIZE
    ITERATIONS = args.ITERATIONS
    NUMREPITER = args.NUMREPITER
    NOISE_SCALE = args.NOISE_SCALE
    OVERRIDE_SAVE = args.OVERRIDE_SAVE
    MODEL_DIRECTORY = args.DIRECTORY

    run_image_metamer_generation(SIDX, LOSS_FUNCTION, INPUTIMAGEFUNCNAME, RANDOMSEED, overwrite_pckl,
                                 use_dataset_preproc, step_size, NOISE_SCALE, ITERATIONS, NUMREPITER,
                                 OVERRIDE_SAVE, MODEL_DIRECTORY)


if __name__ == '__main__':
    main()
