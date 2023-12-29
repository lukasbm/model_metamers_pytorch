import torch as ch


class ImageNet:
    def __init__(self, **kwargs):
        mean = kwargs.get('mean', [0.485, 0.456, 0.406])
        std = kwargs.get('std', [0.229, 0.224, 0.225])

        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor(mean),
            'std': ch.tensor(std),
            'min_value': kwargs.get('min_value', 0),
            'max_value': kwargs.get('max_value', 1),
            'custom_class': None,
            'label_mapping': None,
        }
        self.__dict__.update(ds_kwargs)
