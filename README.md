# TI-ViT

The repository contains script for exporting PyTorch VIT model to ONNX format in the form that is compatible with 
[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) (version 8.6.0.5). 

## Installation

To install export script run the following command:
```commandline
pip3 install git+https://github.com/ENOT-AutoDL/ti-vit.git@main
```

## Examples

### MLP blocks on TI DSP (maximum performance variant)

To export the model version with maximum performance, run the following command:
```commandline
export-ti-vit -o max-perf.onnx -t max-perf
```
This variant of model contains MLP blocks that can be run on TI DSP. GELU operation is approximated.

### MLP blocks partially on TI DSP (minimal loss of accuracy)

To export the model version with minimal loss of accuracy, run the following command:
```commandline
export-ti-vit -o max-acc.onnx -t max-acc
```
This variant of model contains MLP blocks that partially can be run on TI DSP. GELU operation is not approximated.

## Compilation of the exported model

It is important to disable compilation of all nodes except nodes from MLP blocks ("Squeeze" node from MLP must be 
disabled too). The list of operations for ["deny_list:layer_name"](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/08_06_00_05/examples/osrt_python/README.md#options-to-enable-control-on-layer-level-delegation-to-ti-dsparm) 
compiler option can be found in the file "output-onnx-dir/output-onnx-name.deny_list", that is generated with onnx file.

## Results (TI-TDA4-J721EXSKG01EVM)

### TorchVision ViT B16
|          | CPU only | max-acc | max-perf |
|----------|----------|---------|----------|
| Time sec | 3.398    | 2.233   | 1.382    |

### ENOT optimized ViT B16
|          | CPU only | max-acc | max-perf |
|----------|----------|---------|----------|
| Time sec | 0.871    | 0.574   | 0.361    |
