import os

from model_analysis_folders.all_model_info import IMAGENET_PATH, MODEL_BASE_PATH
from robustness import datasets
from robustness.model_utils import make_and_restore_model


def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
        'input_after_preproc',
        #          'relu0',
        'relu0_fake_relu',
        #          'relu1',
        'relu1_fake_relu',
        #          'relu2',
        'relu2_fake_relu',
        #          'relu3',
        'relu3_fake_relu',
        #          'relu4',
        'relu4_fake_relu',
        #          'fc0_relu',
        'fc0_relu_fake_relu',
        #          'fc1_relu',
        'fc1_relu_fake_relu',
        'final'
    ]

    ds = datasets.ImageNet(IMAGENET_PATH)

    ckpt_path = os.path.join(MODEL_BASE_PATH, 'visual_networks', 'pytorch_checkpoints',
                             'alexnet_l2_3_robust_training.pt')
    model, _ = make_and_restore_model(arch='alexnet', dataset=ds,
                                      resume_path=ckpt_path,
                                      pytorch_pretrained=False, parallel=False)

    # send the model to the GPU and return it.
    model.cuda()
    model.eval()
    if return_metamer_layers:
        return model, ds, metamer_layers
    else:
        return model, ds


def main(return_metamer_layers=False,
         ds_kwargs={}):
    if return_metamer_layers:
        model, ds, metamer_layers = build_net(
            return_metamer_layers=return_metamer_layers,
            ds_kwargs=ds_kwargs)
        return model, ds, metamer_layers

    else:
        model, ds = build_net(
            return_metamer_layers=return_metamer_layers,
            ds_kwargs=ds_kwargs)
        return model, ds


if __name__ == "__main__":
    main()
