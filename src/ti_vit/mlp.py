from enum import Enum

import torch
from torch import nn
from torchvision.models.vision_transformer import MLPBlock

from ti_vit.common import copy_weights
from ti_vit.common import sync_device_and_mode


class MLPType(Enum):
    """
    Type of MLP block.

    - CONV_CONV - MLP block assembled as ``convolution_with_kernel_1x1 + activation + convolution_with_kernel_1x1``
    - LINEAR_CONV - MLP block assembled as ``linear + activation + convolution_with_kernel_1x1``

    """

    CONV_CONV = "CONV_CONV"
    LINEAR_CONV = "LINEAR_CONV"


class GeluApproximationType(Enum):
    """
    GELU approximation type.

    - NONE - disable approximation
    - SIGMOID - approximate as ``x * sigmoid(1.702 * x)``
    - TANH - approximate as ``0.5 * x * (tanh(0.7978845834732056 * (x + 0.044715 * x * x * x)) + 1.0)``

    """

    NONE = "NONE"
    SIGMOID = "SIGMOID"
    TANH = "TANH"


class TICompatibleMLP(nn.Module):
    """TI compatible MLP block."""

    def __init__(
        self,
        dims: int,
        hidden_dims: int,
        mlp_type: MLPType = MLPType.CONV_CONV,
        gelu_approx_type: GeluApproximationType = GeluApproximationType.NONE,
    ):
        """
        dims : int
            Number of channels of the input.
        hidden_dims : int
            Number of channels of the expanded tensor.
        mlp_type : MLPType
            MLP type (see ``MLPType`` enum documentation).
        gelu_approx_type : GeluApproximationType
            GELU approximation type (see ``GeluApproximationType`` enum documentation).
        """
        super().__init__()

        try:
            self.gelu = {
                GeluApproximationType.NONE: nn.GELU(),
                GeluApproximationType.SIGMOID: self._gelu_approx_sigmoid,
                GeluApproximationType.TANH: self._gelu_approx_tanh,
            }[gelu_approx_type]
            self._gelu_approx_type = gelu_approx_type
        except KeyError as exc:
            raise ValueError(f'Got unknown type of gelu approximation "{gelu_approx_type}"') from exc

        if mlp_type == MLPType.CONV_CONV:
            self.expand = nn.Conv2d(in_channels=dims, out_channels=hidden_dims, kernel_size=(1, 1))
            self.shrink = nn.Conv2d(in_channels=hidden_dims, out_channels=dims, kernel_size=(1, 1))
        elif mlp_type == MLPType.LINEAR_CONV:
            self.expand = nn.Linear(in_features=dims, out_features=hidden_dims)
            self.shrink = nn.Conv2d(in_channels=hidden_dims, out_channels=dims, kernel_size=(1, 1))
        else:
            raise ValueError(f'Got unknown mlp_type "{mlp_type}"')

        self._mlp_type = mlp_type

    @staticmethod
    def _gelu_approx_tanh(x: torch.Tensor) -> torch.Tensor:
        # This is default torch approximation (0.5 * x * (tanh(0.7978845834732056 * (x + 0.044715 * x * x * x)) + 1.0)),
        # where tanh replaced by (2.0 * nn.functional.sigmoid(2.0 * x) - 1.0)
        return x * torch.sigmoid(1.5957691669464111 * (x + 0.044715 * x * x * x))

    @staticmethod
    def _gelu_approx_sigmoid(x: torch.Tensor) -> torch.Tensor:
        # simplified torch approximation
        return x * torch.sigmoid(1.702 * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        if self._mlp_type == MLPType.CONV_CONV:
            x = x.unsqueeze(3).permute(0, 2, 1, 3)
            x = self.expand(x)
            x = self.gelu(x)
        else:
            x = self.expand(x)
            if self._gelu_approx_type == GeluApproximationType.NONE:
                x = self.gelu(x)
                x = x.unsqueeze(3).permute(0, 2, 1, 3)
            else:
                x = x.unsqueeze(3).permute(0, 2, 1, 3)
                x = self.gelu(x)

        x = self.shrink(x)
        x = x.permute(0, 2, 1, 3).squeeze(3)

        return x

    @classmethod
    def from_module(
        cls,
        vit_mlp: MLPBlock,
        mlp_type: MLPType = MLPType.CONV_CONV,
        gelu_approx_type: GeluApproximationType = GeluApproximationType.NONE,
    ) -> "TICompatibleMLP":
        """
        Create TI compatible MLP block from common ViT MLP block.

        Parameters
        ----------
        vit_mlp : MLPBlock
            Source block.
        mlp_type : MLPType
            MLP type (see ``MLPType`` enum documentation).
        gelu_approx_type : GeluApproximationType
            GELU approximation type (see ``GeluApproximationType`` enum documentation).

        Returns
        -------
        TICompatibleMLP
            Instance of ``TICompatibleMLP`` with appropriate weights, device and training mode.

        """
        expand, shrink = vit_mlp[0], vit_mlp[3]
        if not isinstance(expand, nn.Linear) or not isinstance(shrink, nn.Linear):
            raise ValueError('Got unknown type of vit_mlp. Cannot find "Linear" layers.')
        if not isinstance(vit_mlp[1], nn.GELU):
            raise ValueError('Got unknown type of vit_mlp. Cannot find "GELU" layer.')
        if not isinstance(vit_mlp[2], nn.Dropout) or not isinstance(vit_mlp[4], nn.Dropout):
            raise ValueError('Got unknown type of vit_mlp. Cannot find "dropout" layers.')

        ti_compatible_mlp = cls(
            dims=expand.in_features,
            hidden_dims=expand.out_features,
            mlp_type=mlp_type,
            gelu_approx_type=gelu_approx_type,
        )
        sync_device_and_mode(src=vit_mlp, dst=ti_compatible_mlp)

        copy_weights(src=expand, dst=ti_compatible_mlp.expand)
        copy_weights(src=shrink, dst=ti_compatible_mlp.shrink)

        return ti_compatible_mlp
