import logging
import typing
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional

import torch
from torch import nn
from torchvision.models.vision_transformer import EncoderBlock
from torchvision.models.vision_transformer import VisionTransformer

from ti_vit.attention import AttentionType
from ti_vit.attention import TICompatibleAttention
from ti_vit.mlp import GeluApproximationType
from ti_vit.mlp import MLPType
from ti_vit.mlp import TICompatibleMLP

_LOGGER = logging.getLogger(__name__)


class _BlockCfg(NamedTuple):
    attention_cfg: Optional[Dict[str, Any]]
    mlp_cfg: Optional[Dict[str, Any]]


class _TICompatibleVit(nn.Module):
    def __init__(self, model: VisionTransformer, cfg: Dict[int, _BlockCfg]):
        super().__init__()

        self._model = model

        attn_counter, mlp_counter = 0, 0
        for block_index, block_cfg in cfg.items():
            block: EncoderBlock = typing.cast(EncoderBlock, model.encoder.layers[block_index])

            if block_cfg.attention_cfg is not None:
                self_attention = TICompatibleAttention.from_module(block.self_attention, **block_cfg.attention_cfg)
                setattr(block, "self_attention", self_attention)
                _LOGGER.debug(
                    f"REPLACE {type(block.self_attention)} => {type(self_attention)} "
                    f"(BLOCK={block_index}, CFG={block_cfg.attention_cfg})"
                )
                attn_counter += 1

            if block_cfg.mlp_cfg is not None:
                mlp = TICompatibleMLP.from_module(block.mlp, **block_cfg.mlp_cfg)
                setattr(block, "mlp", mlp)
                _LOGGER.debug(
                    f"REPLACE {type(block.mlp)} => {type(mlp)} " f"(BLOCK={block_index}, CFG={block_cfg.mlp_cfg})"
                )
                mlp_counter += 1

        _LOGGER.info(f"{attn_counter} attentions replaced")
        _LOGGER.info(f"{mlp_counter} MLPs replaced")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return self._model(x)


class TICompatibleVitOrtMaxPerf(_TICompatibleVit):
    """TI compatible Vit model with maximal performance."""

    def __init__(self, model: VisionTransformer, ignore_tidl_errors: bool = False):
        """
        Parameters
        ----------
        model : VisionTransformer
            Source Vit model.
        ignore_tidl_errors : bool
            Experimental option.
        """
        if ignore_tidl_errors:
            cfg = {i: self._mlp_perf_block_cfg() if i < 8 else self._attn_mlp_perf_block_cfg() for i in range(12)}
        else:
            cfg = {i: self._mlp_perf_block_cfg() for i in range(12)}

        super().__init__(model=model, cfg=cfg)

    @staticmethod
    def _attn_mlp_perf_block_cfg() -> _BlockCfg:
        return _BlockCfg(
            attention_cfg={
                "attention_type": AttentionType.CONV_LINEAR,
            },
            mlp_cfg={"mlp_type": MLPType.CONV_CONV, "gelu_approx_type": GeluApproximationType.TANH},
        )

    @staticmethod
    def _mlp_perf_block_cfg() -> _BlockCfg:
        return _BlockCfg(
            attention_cfg=None,
            mlp_cfg={"mlp_type": MLPType.CONV_CONV, "gelu_approx_type": GeluApproximationType.TANH},
        )


class TICompatibleVitOrtMaxAcc(_TICompatibleVit):
    """TI compatible Vit model with minimal accuracy drop."""

    def __init__(self, model: VisionTransformer):
        """
        Parameters
        ----------
        model : VisionTransformer
            Source Vit model.
        """
        super().__init__(
            model=model,
            cfg={i: self._mlp_lc_block_cfg() if i < 8 else self._mlp_cc_block_cfg() for i in range(12)},
        )

    @staticmethod
    def _mlp_lc_block_cfg() -> _BlockCfg:
        return _BlockCfg(
            attention_cfg=None,
            mlp_cfg={"mlp_type": MLPType.LINEAR_CONV, "gelu_approx_type": GeluApproximationType.NONE},
        )

    @staticmethod
    def _mlp_cc_block_cfg() -> _BlockCfg:
        return _BlockCfg(
            attention_cfg=None,
            mlp_cfg={"mlp_type": MLPType.CONV_CONV, "gelu_approx_type": GeluApproximationType.NONE},
        )
