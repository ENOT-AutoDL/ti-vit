import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional
from typing import Union

import torch
from torchvision.models import ViT_B_16_Weights
from torchvision.models import vit_b_16

from ti_vit.model import TICompatibleVitOrtMaxAcc
from ti_vit.model import TICompatibleVitOrtMaxPerf


def export(
    output_onnx_path: Union[str, Path],
    model_type: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
    resolution: int = 224,
) -> None:
    """
    Parameters
    ----------
    output_onnx_path : Union[str, Path]
        Path to the output onnx.
    model_type : str
        Type of the final model. Possible values are "npu-max-acc", "npu-max-perf" or "cpu".
    checkpoint_path : Optional[Union[str, Path]] = None
        Path to the pytorch model checkpoint. If value is None, then ViT_B_16 pretrained torchvision model is used.
        Default value is None.
    resolution : int
        Resolution of input image. Default value is 224.
    """
    if checkpoint_path is None:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT, progress=True)
    else:
        checkpoint = torch.load(str(checkpoint_path))
        model = checkpoint["model_ckpt"]

    model.cpu().eval()

    try:
        transform_model_func = {
            "cpu": lambda model: model,
            "npu-max-acc": TICompatibleVitOrtMaxAcc,
            "npu-max-perf": lambda model: TICompatibleVitOrtMaxPerf(model=model, ignore_tidl_errors=False),
            "npu-max-perf-experimental": lambda model: TICompatibleVitOrtMaxPerf(model=model, ignore_tidl_errors=True),
        }[model_type]
    except KeyError as exc:
        raise ValueError(f"Got unknown transformation type ('{model_type}')") from exc

    model = transform_model_func(model)

    device = next(model.parameters()).device
    dummy_data = torch.ones([1, 3, resolution, resolution], dtype=torch.float32, device=device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # disable export warnings
        torch.onnx.export(
            model=model,
            f=str(output_onnx_path),
            args=dummy_data,
            input_names=["input"],
            output_names=["output"],
            opset_version=9,
        )


def export_ti_compatible_vit() -> None:  # pylint: disable=missing-function-docstring
    logger = logging.getLogger("ti_vit")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-onnx", type=str, required=True, help="Path to the output onnx.")
    parser.add_argument(
        "-t",
        "--model-type",
        type=str,
        required=False,
        default="npu-max-perf",
        help='Type of the final model (optional argument). Possible values are "npu-max-acc", "npu-max-perf", or "cpu".'
        ' Default value is "npu-max-perf".',
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=False,
        help="Path to the Vit checkpoint (optional argument). By default we download the torchvision checkpoint "
        "(VIT_B_16).",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        required=False,
        default=224,
        help="Resolution of input images (optional argument). Default value is 224.",
    )
    args = parser.parse_args()

    export(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output_onnx,
        model_type=args.model_type,
        resolution=args.resolution,
    )
