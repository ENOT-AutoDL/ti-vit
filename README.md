# TI-ViT

The repository contains script for export pytorch VIT model to onnx format in form that compatible with 
[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) (version 8.6.0.5). 

## Installation

To install export script run the following command:
```commandline
pip3 install git+https://github.com/ENOT-AutoDL/ti-vit.git@main
```

## Examples

To export the model version with maximum performance, run the following command:
```commandline
export-ti-vit -o npu-max-perf.onnx -t npu-max-perf
```

To export the model version with minimal loss of accuracy, run the following command:
```commandline
export-ti-vit -o npu-max-acc.onnx -t npu-max-acc
```

