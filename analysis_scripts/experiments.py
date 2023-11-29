from functools import partial

from make_metamers_imagenet_16_category_val_400_only_save_metamer_layers import run_image_metamer_generation
from make_null_distributions import run_null_distribution

model_name = "alexnet"
default_generation = partial(run_image_metamer_generation,
                             image_id=1,
                             loss_func_name="inversion_loss_layer",
                             input_image_func_name="400_16_class_imagenet_val",
                             seed=5,
                             overwrite_pckl=True,
                             use_dataset_preproc=False,
                             step_size=1.0,
                             noise_scale=1 / 20,
                             override_save=True,
                             model_directory=f"model_analysis_folders/visual_networks/{model_name}",
                             # iterations=None,
                             # num_repetitions=None,
                             # initial_metamer=None,
                             # output_name=None,
                             )

# TODO: optimization analysis on every experiment (print_all_distance_measures_and_number_of_metamers)
# note: need to generate null dist first


# run_null_distribution(
#     NUMNULL=50_000, SPLITIDX=0,
#     PATHNULL=f"model_analysis_folders/visual_networks/{model_name}/null_dist/", RANDOMSEED=5,
#     OVERWRITE_PICKLE=True, shuffle=True,
#     MODEL_DIRECTORY=f"model_analysis_folders/visual_networks/{model_name}/"
# )

# print_all_distance_measures_and_number_of_metamers(model_name)
# , use_saved_distances=True, save_metamer_distances=True, )

# experiment
for (iterations, repetitions) in [(100, 10), (100, 100), (1000, 1), (1000, 10), (10000, 1)]:
    print(f"TEST: running initial metamer test (NOISE) with iterations={iterations} and repetitions={repetitions}")
    default_generation(iterations=iterations, num_repetitions=repetitions, initial_metamer="noise",
                       output_name=f"initial_metamer_test_noise_iterations_{iterations}_repetitions_{repetitions}/")

    print(f"TEST: running initial metamer test (REFERENCE) with iterations={iterations} and repetitions={repetitions}")
    default_generation(iterations=iterations, num_repetitions=repetitions, initial_metamer="reference",
                       output_name=f"initial_metamer_test_reference_iterations_{iterations}_repetitions_{repetitions}/")

    print(f"TEST: running initial metamer test (UNIFORM) with iterations={iterations} and repetitions={repetitions}")
    default_generation(iterations=iterations, num_repetitions=repetitions, initial_metamer="uniform",
                       output_name=f"initial_metamer_test_uniform_iterations_{iterations}_repetitions_{repetitions}/")
