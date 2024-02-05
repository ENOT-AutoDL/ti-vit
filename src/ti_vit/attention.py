import typing
from enum import Enum
from typing import Tuple

import torch
from torch import nn

from ti_vit.common import copy_weights
from ti_vit.common import sync_device_and_mode


class AttentionType(Enum):
    """
    Type of attention block.

    - CONV_CONV - qkv projection and output projection is a convolution with 1x1 kernel
    - CONV_LINEAR - qkv projection is a convolution with 1x1 kernel, output projection is linear

    """

    CONV_CONV = "CONV_CONV"
    CONV_LINEAR = "CONV_LINEAR"


class TICompatibleAttention(nn.Module):
    """TI compatible attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attention_type: AttentionType = AttentionType.CONV_LINEAR,
    ):
        """
        Parameters
        ----------
        dim : int
            Total dimension of the model.
        num_heads : int
            Number of parallel attention heads.
        qkv_bias : bool
            If True, adds a learnable bias to the qkv projection. Default value is False.
        attention_type : AttentionType
            Type of attention block (see ``AttentionType`` enum documentation).
        """
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f'"dim"={dim} should be divisible by "num_heads"={num_heads}')

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        if attention_type == AttentionType.CONV_CONV:
            self.qkv_proj = nn.Conv2d(in_channels=dim, out_channels=dim * 3, kernel_size=(1, 1), bias=qkv_bias)
            self.out_proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 1))
        elif attention_type == AttentionType.CONV_LINEAR:
            self.qkv_proj = nn.Conv2d(in_channels=dim, out_channels=dim * 3, kernel_size=(1, 1), bias=qkv_bias)
            self.out_proj = nn.Linear(in_features=dim, out_features=dim)
        else:
            raise ValueError(f'Got unknown attention_type "{attention_type}"')

        self._attention_type = attention_type

    def forward(  # pylint: disable=missing-function-docstring
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, None]:
        del key, value

        assert not need_weights

        x = query
        B, N, C = x.shape

        # (B, N, C) -> (B, N, C, 1) -> (B, C, N, 1)
        x = x.unsqueeze(3).permute(0, 2, 1, 3)

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, 3, C, N)
        q, k, v = qkv.split(1, dim=1)

        # (B, 1, C, N) -> (B, H, C//H, N) -> (B, H, N, C//H)
        q = q.reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)
        # (B, 1, C, N) -> (B, H, C//H, N)
        k = k.reshape(B, self.num_heads, C // self.num_heads, N)
        # (B, 1, C, N) -> (B, H, C//H, N) -> (B, H, N, C//H)
        v = v.reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v

        if self._attention_type == AttentionType.CONV_CONV:
            # (B, H, N, C//H) -> (B, H, C//H, N) -> (B, C, N, 1)
            x = x.permute(0, 1, 3, 2).reshape(B, C, N, 1)
            x = self.out_proj(x)
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(3)
        else:
            # (B, H, N, C//H) -> (B, N, H, C//H) -> (B, N, C)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)
            x = self.out_proj(x)

        return x, None

    @classmethod
    def from_module(
        cls,
        vit_attn: nn.Module,
        attention_type: AttentionType = AttentionType.CONV_CONV,
    ) -> "TICompatibleAttention":
        """
        Create TI compatible attention block from common ViT attention block.

        Parameters
        ----------
        vit_attn : nn.Module
            Source block.
        attention_type : AttentionType
            Attention type (see ``AttentionType`` enum documentation).

        Returns
        -------
        TICompatibleAttention
            Instance of ``TICompatibleAttention`` with appropriate weights, device and training mode.

        """
        if hasattr(vit_attn, "qkv"):
            qkv_proj = typing.cast(nn.Linear, vit_attn.qkv)
            out_proj = typing.cast(nn.Linear, vit_attn.proj)
        else:
            in_proj_weight = typing.cast(nn.Parameter, vit_attn.in_proj_weight)
            out_features, in_features = in_proj_weight.shape
            qkv_proj = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=hasattr(vit_attn, "in_proj_bias"),
                device=in_proj_weight.device,
                dtype=in_proj_weight.dtype,
            )
            qkv_proj.weight = in_proj_weight
            qkv_proj.bias = vit_attn.in_proj_bias  # pyright: ignore[reportAttributeAccessIssue]

            out_proj = typing.cast(nn.Linear, vit_attn.out_proj)

        ti_compatible_attn = cls(
            dim=qkv_proj.in_features,
            num_heads=typing.cast(int, vit_attn.num_heads),
            qkv_bias=qkv_proj.bias is not None,
            attention_type=attention_type,
        )
        sync_device_and_mode(src=vit_attn, dst=ti_compatible_attn)

        copy_weights(src=qkv_proj, dst=ti_compatible_attn.qkv_proj)
        copy_weights(src=out_proj, dst=ti_compatible_attn.out_proj)

        if hasattr(vit_attn, "scale"):
            ti_compatible_attn.scale = vit_attn.scale

        return ti_compatible_attn
