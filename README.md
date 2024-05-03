# CNNs Quantization Based on Small Logarithmic Floating-Point Number

## Project Description

This project provides a PyTorch implementation of quantized convolutional neural networks that are easy to deploy on hardware. It supports 8-bit small logarithmic floating point (SLFP) and 7-bit small floating point (SFP) quantization, based on max-scaling quantization. The quantization details can be found in[ref 1](https://ieeexplore.ieee.org/document/9920192).

The project has been tested on the CIFAR-100 and ImageNet-1k datasets, and currently supports MobileNet, VGG-16, ResNet-50, ShuffleNet V2, AlexNet, and SqueezeNet1.0.

## Key Contributions

1. SLFP<3,4> quantization. Results in a post-training quantization accuracy loss of less than 1% in most CNNs.
2. Revise SGD optimizer for non-uniform quantization. Fine-tuning the quantized model with 6% of the training set can effectively restore model accuracy.
3. Model optimization. By optimizing the activation functions in the network, model performance is further improved.
4. After model optimization, 8-bit SLFP quantization and retraining, the accuracy of the quantized models can exceed the FP32 baseline.

## Installation and Running

1. Clone this repository
2. Run the code: `python ./cifar100_train_eval.py --Qbits <bit width> --net <net name> ...` or `python ./imgnet_train_eval.py --Qbits <bit width> --net <net name> ...`
Arguments are optional, please refer to the argparse settings in the code. The default setting is 32-bit floating point reference of MobileNetV1 on CIFAR-100 or ImageNet.

