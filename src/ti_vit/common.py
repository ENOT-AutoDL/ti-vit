from typing import Union

import torch
from torch import nn


def copy_weights(src: nn.Linear, dst: Union[nn.Linear, nn.Conv2d]) -> None:
    """
    Update weights and bias parameters of the destination module with values from the source module.

    Parameters
    ----------
    src : nn.Linear
        The source module.
    dst : Union[nn.Linear, nn.Conv2d]
        The destination module.

    """
    with torch.no_grad():
        if isinstance(dst, nn.Linear):
            dst.weight.copy_(src.weight)
        elif isinstance(dst, nn.Conv2d):
            dst.weight.copy_(src.weight.unsqueeze(-1).unsqueeze(-1))
        else:
            raise TypeError(f"dst must be nn.Linear or nn.Conv2d (type(dst)={type(dst)})")

        if src.bias is not None:
            dst.bias.copy_(src.bias)  # pyright: ignore[reportOptionalMemberAccess]


def sync_device_and_mode(src: nn.Module, dst: nn.Module) -> None:
    """
    Update device and training mode parameters of the destination module with values from the source module.

    Parameters
    ----------
    src : nn.Module
        The source module.
    dst : nn.Module
        The destination module.

    """
    device = next(src.parameters()).device
    dst.to(device=device)
    dst.train(mode=src.training)
