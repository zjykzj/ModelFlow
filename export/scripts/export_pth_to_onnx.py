# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 14:10
@File    : export_pth_to_onnx.py
@Author  : zj
@Description: 

Script for exporting a PyTorch model to ONNX format.

This script supports:
- Exporting selected torchvision classification models (e.g., resnet, mobilenet)
- Using local pre-trained PyTorch model files
- Toy model support via '--model_name toy'
- Custom input shape and number of output classes
- Dynamic batch size support in ONNX export

"""

import os
import sys
import argparse

# 将当前工作目录加入 Python 模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.onnx_exporter import export_to_onnx, SUPPORTED_MODELS


def parse_args():
    """
    Parse command-line arguments for exporting a PyTorch model to ONNX.

    Supported models include:
        - ResNet series: resnet18, resnet34, ..., resnet152
        - MobileNet series: mobilenet_v2, mobilenet_v3_small/large
        - ShuffleNet series: shufflenet_v2_x0_5, shufflenet_v2_x1_0
        - EfficientNet series: efficientnet_b0, b1, b2
        - Toy model: toy (for testing)

    Example usage:
        python export_pth_to_onnx.py --model_name resnet50 --save_path resnet50.onnx
        python export_pth_to_onnx.py --model_name toy --save_path toy.onnx
        python export_pth_to_onnx.py --model_name mobilenet_v2 --pretrain_path ./model.pth --save_path mobilenet.onnx
    """
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")

    # Model source
    parser.add_argument("model_name", type=str,
                        choices=SUPPORTED_MODELS,
                        help=f"Model name. Choose from: {SUPPORTED_MODELS}")

    parser.add_argument("--pretrain_path", type=str,
                        default=None,
                        help="Path to local pre-trained .pt/.pth file. "
                             "If not specified, uses torchvision's pretrained weights.")

    # Output configuration
    parser.add_argument("--save_path", type=str, required=True,
                        help="Output path for the exported ONNX model. Example: '../models/resnet50.onnx'")

    # ONNX export settings
    parser.add_argument('--opset', type=int, default=12,
                        help="ONNX opset version used during export. Default: 12")

    parser.add_argument("--dynamic", action="store_true",
                        help="Enable dynamic batch size in the exported ONNX model")

    # Model configuration
    parser.add_argument("--num_classes", type=int, default=1000,
                        help="Number of output classes. Default is 1000 for ImageNet-trained models.")

    parser.add_argument("--input_shape", nargs='+', type=int, default=None,
                        help="Input shape as [C H W] (without batch dimension). Example: 3 224 224")

    args = parser.parse_args()

    print("Arguments parsed:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    return args


def main():
    args = parse_args()

    export_to_onnx(
        model_name=args.model_name,
        pretrain_path=args.pretrain_path,
        save_path=args.save_path,
        opset=args.opset,
        dynamic_axes=args.dynamic,
        num_classes=args.num_classes,
        input_shape=args.input_shape,
    )


if __name__ == "__main__":
    main()
