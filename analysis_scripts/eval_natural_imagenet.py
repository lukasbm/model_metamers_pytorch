"""
loads a network (based on build_network.py) and evaluates it on imagenet.
Output statistics are top1 accuracy, loss and time for one epoch.
This does not do adversarial evaluation, just natural evaluation.
"""
import argparse
import importlib.util
import os
from pprint import pprint

import numpy as np
import torch as ch
from cox import store
from cox.utils import Parameters

from robustness import train, defaults
from robustness.datasets import ImageNet


def run_imagenet_natural_eval(RANDOMSEED, MODEL_DIRECTORY, BATCH_SIZE, NUM_WORKERS):
    # battling the python import system
    if MODEL_DIRECTORY is None:
        import build_network
        MODEL_DIRECTORY = ''  # use an empty string to append to saved files.
    else:
        build_network_spec = importlib.util.spec_from_file_location("build_network",
                                                                    os.path.join(MODEL_DIRECTORY, 'build_network.py'))
        build_network = importlib.util.module_from_spec(build_network_spec)
        build_network_spec.loader.exec_module(build_network)

    model, ds = build_network.main()

    ch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)

    train_loader, val_loader = ds.make_loaders(
        batch_size=BATCH_SIZE,
        workers=NUM_WORKERS,
        shuffle_train=False,
        shuffle_val=False,
        data_aug=True,
    )

    # Hard-coded base parameters
    eval_kwargs = {
        'out_dir': os.path.join(MODEL_DIRECTORY, "eval_out"),
        'exp_name': "eval_natural_imagenet",
        'adv_train': 0,  # don't do adversarial training
        "adv_eval": 0,  # don't do adversarial evaluation
        'constraint': '2',
        'eps': 3,
        'step_size': 1,
        'attack_lr': 1.5,
        'attack_steps': 20,
        'save_ckpt_iters': 1,
    }
    eval_args = Parameters(eval_kwargs)
    # Fill whatever parameters are missing with the defaults
    eval_args = defaults.check_and_fill_args(eval_args, defaults.TRAINING_ARGS, ImageNet)
    eval_args = defaults.check_and_fill_args(eval_args, defaults.PGD_ARGS, ImageNet)

    # Create the cox store, and save the arguments in a table
    store_out = store.Store(eval_args.out_dir, eval_args.exp_name)
    print(store_out)
    args_dict = eval_args.as_dict() if isinstance(eval_args, Parameters) else vars(eval_args)
    # schema = store.schema_from_dict(args_dict)
    store_out.add_table_like_example('metadata', args_dict)
    store_out['metadata'].append_row(args_dict)

    # run the actual evaluation
    log_info = train.eval_model(eval_args, model, val_loader, store=store_out)
    print("evaluation complete, these are the results:")
    pprint(log_info)


def main(raw_args=None):
    #########PARSE THE ARGUMENTS FOR THE FUNCTION#########
    parser = argparse.ArgumentParser(description='Input the arguments for imagenet evaluation')
    parser.add_argument('-R', '--RANDOMSEED', metavar='--R', type=int, default=0,
                        help='random seed to use for synthesis')
    parser.add_argument('-B', '--BATCH_SIZE', metavar='--B', type=int, default=64, help='batch size for evaluation')
    parser.add_argument('-N', '--NUM_WORKERS', metavar='--N', type=int, default=8,
                        help='number of workers for data loading')
    parser.add_argument('-D', '--DIRECTORY', metavar='--D', type=str, default=None,
                        help='The directory with the location of the `build_network.py` file. Folder structure for saving metamers will be created in this directory. If not specified, assume this script is located in the same directory as the build_network.py file.')

    args = parser.parse_args(raw_args)

    RANDOMSEED = args.RANDOMSEED
    MODEL_DIRECTORY = args.DIRECTORY
    BATCH_SIZE = args.BATCH_SIZE
    NUM_WORKERS = args.NUM_WORKERS

    run_imagenet_natural_eval(RANDOMSEED, MODEL_DIRECTORY, BATCH_SIZE, NUM_WORKERS)


if __name__ == '__main__':
    main()
