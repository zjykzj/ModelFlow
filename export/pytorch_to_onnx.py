# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/13 15:08
@File    : pytorch_to_onnx.py
@Author  : zj
@Description:

>>>python3 pytorch_to_onnx.py --model resnet18 --save resnet18.onnx --img-size 224 --img-channels 3
>>>python3 pytorch_to_onnx.py --model efficientnet_b0 --save efficientnet_b0.onnx --img-size 224 --img-channels 3

"""

import os
import argparse
import numpy as np
from packaging import version

import onnx
import onnxruntime

import torch
import torchvision
import torch.nn as nn
from torchvision import models


# ==================== 参数解析 ====================
def parse_opt():
    parser = argparse.ArgumentParser(description="PyTorch to ONNX Converter (Fixed Batch=1)")
    parser.add_argument("--model", type=str, default=None,
                        help="PyTorch model name from torchvision.models (e.g., resnet18) or None for custom/toy model")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to custom trained .pth/.pt model weights. Required if --model is not a torchvision model.")
    parser.add_argument("--save", type=str, required=True,
                        help="Path to save the ONNX model, e.g., '../assets/model.onnx'")
    parser.add_argument("--img-size", type=int, nargs='+', default=[224],
                        help="Input image size: specify one int for square (e.g., 224) or two for H W (e.g., 224 224)")
    parser.add_argument("--img-channels", type=int, default=3,
                        help="Number of input image channels (1 for grayscale, 3 for RGB), default: 3")
    parser.add_argument("--dummy-labels", action="store_true",
                        help="Forward with dummy labels (for models requiring labels, e.g., some segmentation heads)")

    args = parser.parse_args()

    # 处理 img_size
    if len(args.img_size) == 1:
        args.img_size = (args.img_size[0], args.img_size[0])
    elif len(args.img_size) == 2:
        args.img_size = tuple(args.img_size)
    else:
        raise ValueError("--img-size must have 1 or 2 values")

    print(f"Arguments: {args}")
    return args


# ==================== 模型加载 ====================
def load_model(args) -> torch.nn.Module:
    """加载模型：支持 torchvision 预定义模型，兼容新旧版本，无警告"""
    if args.model is None and args.weights is None:
        raise ValueError("Either --model or --weights must be provided.")

    if args.model and args.model in models.__dict__:
        print(f"Loading torchvision model: {args.model}")

        model_entry = models.__dict__[args.model]
        tv_version = version.parse(torchvision.__version__)

        if tv_version >= version.parse("0.13"):
            # --- 构造正确的 Weights 类名 ---
            # 特殊模型映射表（最准确的方式）
            special_model_mapping = {
                'efficientnet_b0': 'EfficientNet_B0_Weights',
                'efficientnet_b1': 'EfficientNet_B1_Weights',
                'efficientnet_b2': 'EfficientNet_B2_Weights',
                'efficientnet_b3': 'EfficientNet_B3_Weights',
                'efficientnet_b4': 'EfficientNet_B4_Weights',
                'efficientnet_b5': 'EfficientNet_B5_Weights',
                'efficientnet_b6': 'EfficientNet_B6_Weights',
                'efficientnet_b7': 'EfficientNet_B7_Weights',
                'efficientnet_v2_s': 'EfficientNet_V2_S_Weights',
                'efficientnet_v2_m': 'EfficientNet_V2_M_Weights',
                'efficientnet_v2_l': 'EfficientNet_V2_L_Weights',
                'mobilenet_v2': 'MobileNet_V2_Weights',
                'mobilenet_v3_large': 'MobileNet_V3_Large_Weights',
                'mobilenet_v3_small': 'MobileNet_V3_Small_Weights',
                'mnasnet0_5': 'MNASNet0_5_Weights',
                'mnasnet0_75': 'MNASNet0_75_Weights',
                'mnasnet1_0': 'MNASNet1_0_Weights',
                'mnasnet1_3': 'MNASNet1_3_Weights',
                'shufflenet_v2_x0_5': 'ShuffleNet_V2_X0_5_Weights',
                'shufflenet_v2_x1_0': 'ShuffleNet_V2_X1_0_Weights',
                'shufflenet_v2_x1_5': 'ShuffleNet_V2_X1_5_Weights',
                'shufflenet_v2_x2_0': 'ShuffleNet_V2_X2_0_Weights',
                'squeezenet1_0': 'SqueezeNet1_0_Weights',
                'squeezenet1_1': 'SqueezeNet1_1_Weights',
            }

            # 优先使用映射表
            if args.model in special_model_mapping:
                weights_name = special_model_mapping[args.model]
            elif '_' in args.model:
                # 通用规则：分割后每部分 capitalize，并用下划线连接
                parts = args.model.split('_')
                capitalized = '_'.join([part.capitalize() for part in parts])
                weights_name = f"{capitalized}_Weights"
            else:
                # 无下划线的模型（如 resnet18）
                if args.model.startswith('resnet'):
                    num = args.model[6:]
                    weights_name = f"ResNet{num}_Weights"
                elif args.model.startswith('resnext'):
                    num = args.model[7:]
                    weights_name = f"ResNeXt{num}_Weights"
                elif args.model.startswith('wide_resnet'):
                    num = args.model[11:]
                    weights_name = f"Wide_ResNet{num}_Weights"
                else:
                    weights_name = f"{args.model.capitalize()}_Weights"

            # 检查是否存在
            if hasattr(models, weights_name):
                weights_cls = getattr(models, weights_name)
                weights = weights_cls.IMAGENET1K_V1
                print(f"Using weights: {weights} (torchvision {tv_version})")
                model = model_entry(weights=weights)
            else:
                raise ValueError(
                    f"No weights enum '{weights_name}' found for '{args.model}'. "
                    f"Available: {[k for k in dir(models) if k.endswith('_Weights')]}"
                )
        else:
            # 旧版本 torchvision
            print(f"Using pretrained=True (torchvision {tv_version})")
            model = model_entry(pretrained=True)

    elif args.weights is not None:
        raise NotImplementedError("Custom model loading not implemented.")
    else:
        raise ValueError(f"Model '{args.model}' not found in torchvision.models")

    model.eval()
    return model


# ==================== ONNX 验证与输出检查 ====================
def check_onnx(onnx_path: str):
    """检查 ONNX 模型格式是否合法"""
    print(f"Validating ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model check passed.")


def check_output(torch_model: nn.Module, onnx_path: str, input_tensor: torch.Tensor):
    """对比 PyTorch 与 ONNX 的输出"""
    print("Checking outputs between PyTorch and ONNXRuntime...")

    # 设置 ONNX Runtime session
    try:
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"ONNX Runtime failed to load: {e}")
        return

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # PyTorch 推理
    with torch.no_grad():
        torch_outputs = torch_model(input_tensor)
    if not isinstance(torch_outputs, (list, tuple)):
        torch_outputs = [torch_outputs]

    # ONNX 推理
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)

    # 对比每个输出
    for i, (torch_out, ort_out) in enumerate(zip(torch_outputs, ort_outs)):
        try:
            np.testing.assert_allclose(to_numpy(torch_out), ort_out, rtol=1e-03, atol=1e-05)
            print(f"✅ Output {i} matches (tolerance: rtol=1e-3, atol=1e-5)")
        except AssertionError as e:
            print(f"❌ Output {i} mismatch: {e}")


# ==================== 导出为 ONNX ====================
def export_to_onnx(torch_model: nn.Module, onnx_path: str, img_channels: int, img_size: tuple):
    """将 PyTorch 模型导出为 ONNX，支持单输入多输出"""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # 创建输入张量 (batch_size=1)
    height, width = img_size
    dummy_input = torch.randn(1, img_channels, height, width, requires_grad=False)

    # 运行一次前向传播，以确定输出结构
    with torch.no_grad():
        output = torch_model(dummy_input)

    # 处理输出：统一为 list 形式
    if isinstance(output, (list, tuple)):
        output_names = [f"output{i}" for i in range(len(output))]
    else:
        output_names = ["output0"]
        output = [output]

    print(f"Exporting model with input shape: {dummy_input.shape}")
    print(f"Detected {len(output)} outputs -> names: {output_names}")

    # 导出 ONNX
    torch.onnx.export(
        model=torch_model,
        args=dummy_input,
        f=onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["image"],  # 固定输入名为 'image'
        output_names=output_names,  # 动态生成 output0, output1, ...
        dynamic_axes=None,  # 当前不启用动态 batch
        verbose=False
    )

    print(f"ONNX model exported to: {onnx_path}")

    # 验证并对比输出
    check_onnx(onnx_path)
    check_output(torch_model, onnx_path, dummy_input)


# ==================== 主函数 ====================
def main(args):
    # 加载模型
    model = load_model(args)
    print(f"Model loaded: {type(model).__name__}")

    # 导出 ONNX
    export_to_onnx(
        torch_model=model,
        onnx_path=os.path.abspath(args.save),
        img_channels=args.img_channels,
        img_size=args.img_size
    )


if __name__ == "__main__":
    args = parse_opt()
    main(args)
