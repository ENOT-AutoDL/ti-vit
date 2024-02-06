import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional
from typing import Union

import onnx
import torch
from onnxsim import simplify
from torchvision.models import ViT_B_16_Weights
from torchvision.models import vit_b_16

from ti_vit.model import TICompatibleVitOrtMaxAcc
from ti_vit.model import TICompatibleVitOrtMaxPerf

_LOGGER = logging.getLogger(__name__)


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
        Path to the PyTorch model checkpoint. If value is None, then ViT_B_16 pretrained torchvision model is used.
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

    output_onnx_path = Path(output_onnx_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # disable export warnings
        torch.onnx.export(
            model=model,
            f=str(output_onnx_path.resolve()),
            args=dummy_data,
            input_names=["input"],
            output_names=["output"],
            opset_version=9,
        )
        _LOGGER.info(f'model exported to onnx (path = "{output_onnx_path}")')

    onnx_model = onnx.load(output_onnx_path)
    onnx_model, ok = simplify(onnx_model)
    if not ok:
        _LOGGER.error("onnx-simplifier step is failed")
    else:
        onnx.save_model(onnx_model, f=output_onnx_path)
        _LOGGER.info("onnx simplified")

    if model_type != "cpu":
        deny_list = [node.name for node in onnx_model.graph.node if "mlp" not in node.name or node.op_type == "Squeeze"]
        deny_list_path = output_onnx_path.with_suffix(".deny_list")
        with deny_list_path.open("wt") as deny_list_file:  # pylint: disable=unspecified-encoding
            json.dump(deny_list, fp=deny_list_file, indent=4)
            _LOGGER.info(f'deny list created (path = "{output_onnx_path}")')


def export_ti_compatible_vit() -> None:  # pylint: disable=missing-function-docstring
    logger = logging.getLogger("ti_vit")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt="%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
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
        help="Path to the ViT checkpoint (optional argument). By default torchvision checkpoint is downloaded."
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
