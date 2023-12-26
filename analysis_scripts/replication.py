from functools import partial

from make_metamers_imagenet_16_category_val_400_only_save_metamer_layers import run_image_metamer_generation

default_generation = partial(run_image_metamer_generation,
                             loss_func_name="inversion_loss_layer",
                             input_image_func_name="400_16_class_imagenet_val",
                             seed=0,
                             overwrite_pckl=True,
                             use_dataset_preproc=False,  # has nothing to with input normalization
                             initial_step_size=1.0,
                             noise_scale=1 / 20,  # noise mean will be 0.5
                             override_save=True,
                             iterations=1000,
                             num_repetitions=8,
                             initial_metamer="noise",
                             # model_directory=f"model_analysis_folders/visual_networks/{model_name}",
                             # image_id=1,
                             # output_name=None,
                             )


def generate(image_id, model_name, output_name):
    default_generation(
        image_id=image_id,
        model_directory=f"model_analysis_folders/visual_networks/{model_name}",
        output_name=output_name
    )


if __name__ == "__main__":
    generate(image_id=32, model_name="alexnet", output_name="32_alexnet_standard")
    generate(image_id=32, model_name="resnet50", output_name="32_resnet50_standard")
