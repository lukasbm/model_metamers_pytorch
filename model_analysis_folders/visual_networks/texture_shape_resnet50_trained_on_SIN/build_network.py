from model_analysis_folders.all_model_info import IMAGENET_PATH
from robustness import datasets
from robustness.model_utils import make_and_restore_model


def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
        'input_after_preproc',
        #          'conv1_relu1',
        'conv1_relu1_fake_relu',
        #          'layer1',
        'layer1_fake_relu',
        #          'layer2',
        'layer2_fake_relu',
        #          'layer3',
        'layer3_fake_relu',
        #          'layer4',
        'layer4_fake_relu',
        'avgpool',
        'final'
    ]

    ds = datasets.ImageNet(IMAGENET_PATH)

    model, _ = make_and_restore_model(arch='texture_shape_resnet50_trained_on_SIN', dataset=ds,
                                      pytorch_pretrained=True, parallel=False)

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
