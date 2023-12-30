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
from typing import Optional

import numpy as np
import torch
import torchvision.models.feature_extraction as tv_fe

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


def load_image_257() -> torch.Tensor:
    image_path = "/home/lukas/Documents/uni/feathers_model_metamers_pytorch/assets/full_400_16_class_imagenet_val_images/257_10_dog_n02085782_00031965.JPEG"
    from torchvision.io import ImageReadMode
    import torchvision
    import torchvision.transforms.v2 as tv
    image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
    transform = tv.Compose([
        tv.CenterCrop(224),
        tv.ToDtype(torch.float, scale=True),
    ])
    image = transform(image)
    assert image.max() <= 1.0
    image = image.unsqueeze(0).float().contiguous().cuda()
    return image


def load_image_257_2() -> torch.Tensor:
    image_path = "/home/lukas/Documents/uni/feathers_model_metamers_pytorch/assets/full_400_16_class_imagenet_val_images/257_10_dog_n02085782_00031965.JPEG"
    from torchvision.io import ImageReadMode
    import torchvision
    import torchvision.transforms.v2 as tv
    image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
    transform = tv.Compose([
        tv.CenterCrop(min(image.shape[-1], image.shape[-2])),
        tv.Resize(224),
        # tv.ToDtype(torch.float, scale=True),
    ])
    image = transform(image)
    image = image.float() / 255
    assert image.max() <= 1.0
    image = image.unsqueeze(0).contiguous()
    return image


def load_image_257_3() -> torch.Tensor:
    import torchvision.transforms.v2 as tv
    from PIL import Image

    image_path = "/home/lukas/Documents/uni/feathers_model_metamers_pytorch/assets/full_400_16_class_imagenet_val_images/257_10_dog_n02085782_00031965.JPEG"

    img_pil = Image.open(image_path)
    im_shape = 224
    transformations = tv.Compose([
        tv.CenterCrop(min(img_pil.size)),  # Crop the image to a square
        tv.Resize((im_shape, im_shape), antialias=True),  # Resize the image to 224x224
        tv.PILToTensor(),
    ])
    img_tensor = transformations(img_pil)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.contiguous()


def load_image_257_orig() -> torch.Tensor:
    from PIL import Image

    image_path = "/home/lukas/Documents/uni/feathers_model_metamers_pytorch/assets/full_400_16_class_imagenet_val_images/257_10_dog_n02085782_00031965.JPEG"
    img_pil = Image.open(image_path)
    width, height = img_pil.size

    # do a square crop
    smallest_dim = min((width, height))
    left = (width - smallest_dim) // 2
    right = (width + smallest_dim) // 2
    top = (height - smallest_dim) // 2
    bottom = (height + smallest_dim) // 2

    im_shape = 224
    img_pil = img_pil.crop((left, top, right, bottom))
    img_pil = img_pil.resize((im_shape, im_shape))
    img_pil.load()
    img1 = np.asarray(img_pil, dtype="float32")
    img1 = np.rollaxis(np.array(img1), 2, 0)
    img1 = img1 / 255
    return torch.tensor(np.expand_dims(img1, 0)).float().contiguous()


def run_image_metamer_generation(output_name: Optional[str] = None):
    iterations = 1000
    overwrite_pckl = True
    num_repetitions = 8
    model_directory = f"model_analysis_folders/visual_networks/alexnet"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preproc = InputNormalize(
        new_mean=torch.tensor([0.485, 0.456, 0.406], device=device),
        new_std=torch.tensor([0.229, 0.224, 0.225], device=device),
        min_value=0.0,
        max_value=1.0,
    )

    # init model (and feature extractor)
    layer_to_invert = "features.9"  # relu3
    class_model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
    fe_model = SingleFeatureExtractor(class_model, layer_to_invert)
    ds = ImageNet()
    model = AttackerModel(fe_model, ds, preproc=preproc)
    assert isinstance(model, torch.nn.Module), "model is no valid torch module"
    assert type(model) is AttackerModel, "model is no valid robustness attacker model"

    # LOAD REFERENCE IMAGE
    reference_image = load_image_257_3()  # load_image(image_id)

    print(reference_image.shape, reference_image.min(), reference_image.max(), reference_image.mean())

    # Set up the saving folder and make sure that the file doesn't already exist
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
