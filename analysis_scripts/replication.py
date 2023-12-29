import os
import random
from typing import Optional

import numpy as np
import torch

from make_metamers_imagenet_16_category_val_400_only_save_metamer_layers import \
    run_image_metamer_generation as metamer_generation_original
from replicate.generate import run_image_metamer_generation as metamer_generation_simple


def setup_pytorch(seed: Optional[int] = None) -> torch.device:
    """
    sets up pytorch with all kinds of settings and performance optimizations

    copied from mad-project
    with most information from: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True

    # set seeds for reproducibility
    if seed is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # some CudNN operations/solvers are not deterministic even with a fixed seed.
        # force usage of deterministic implementations when a seed is set.
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # benchmark chooses the best algorithm based on a heuristic
        # especially useful when input sizes do not change
        # https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054/2
        torch.backends.cudnn.benchmark = True

    # set data types so that we can use tensor cores
    # enable cuda data type (the regular 32bit float can not run on tensor cores)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # use oneDNN graph with TorchScript for inference
    torch.jit.enable_onednn_fusion(True)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_original(image_id, model_name, output_name, seed):
    metamer_generation_original(
        loss_func_name="inversion_loss_layer",
        input_image_func_name="400_16_class_imagenet_val",
        overwrite_pckl=True,
        use_dataset_preproc=False,  # has nothing to with input normalization
        initial_step_size=1.0,
        noise_scale=1 / 20,  # noise mean will be 0.5
        override_save=True,
        iterations=1000,
        num_repetitions=8,
        initial_metamer="noise",
        fake_relu=False,
        image_id=image_id,
        model_directory=f"model_analysis_folders/visual_networks/{model_name}",
        output_name=output_name,
        seed=seed
    )


def generate_simple(output_name):
    metamer_generation_simple(
        output_name=output_name,
        image_id=257,
    )


if __name__ == "__main__":
    seed = 0
    setup_pytorch(seed=seed)
    # generate(image_id=32, model_name="alexnet", output_name="32_alexnet_standard", seed=seed)
    # generate(image_id=257, model_name="alexnet", output_name="257_alexnet_standard", seed=seed)
    # generate(image_id=257, model_name="alexnet", output_name="257_alexnet_standard_my_fe_model_relu3_minimal", seed=seed)
    # generate(image_id=32, model_name="resnet50", output_name="32_resnet50_standard", seed=seed)

    generate_simple("257_alexnet_standard_simplest")
    # generate_original(image_id=257, model_name="alexnet", output_name="257_alexnet_standard_original", seed=seed)
