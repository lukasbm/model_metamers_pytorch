import torch


class InversionLossLayerReplica(torch.nn.Module):
    """Loss used for most metamer generation experiments"""

    def __init__(self, normalize_loss=False):
        super().__init__()
        self.normalize_loss = normalize_loss

    def forward(self, model, inp, targ):
        activations = model(inp)
        rep = activations.contiguous().view(activations.size(0), -1)
        if self.normalize_loss:
            loss = torch.div(
                torch.norm(rep - targ, dim=1),
                torch.norm(targ, dim=1)
            )
        else:
            loss = torch.norm(rep - targ, dim=1)
        return loss, None
