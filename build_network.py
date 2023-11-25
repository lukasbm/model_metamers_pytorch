"""
just a simple script that allows for default usage of alex net when building/evaluating models.

Feel free to remove this file if no longer needed.
It just serves to easy the exploration process and to avoid errors on all the mystical `import build_network` calls.
"""

from model_analysis_folders.visual_networks.alexnet.build_network import build_net


def main(return_metamer_layers=False,
         ds_kwargs=None):
    if ds_kwargs is None:
        ds_kwargs = {}
    return build_net(
        return_metamer_layers=return_metamer_layers,
        ds_kwargs=ds_kwargs
    )
