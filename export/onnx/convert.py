# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : convert.py
@Author  : zj
@Description: torchvision 分类模型 ONNX 导出器

支持 ResNet, EfficientNet, MobileNet, ShuffleNet, SqueezeNet,
MNASNet, DenseNet, VGG, ConvNeXt, ViT 等主流分类模型。

典型用法：
    >>> from export.onnx import TorchvisionExporter
    >>> exporter = TorchvisionExporter("efficientnet_b0")
    >>> onnx_path = exporter.export_onnx("model.onnx", img_size=224)
"""

import os
import argparse
from typing import Optional, Tuple, Union

from packaging import version

import onnx
import onnxruntime
import torch
import torch.nn as nn
from torchvision import models

from export.core.base import BaseExporter
from export.core.validation import check_onnx, compare_output


# ==================== torchvision 版本兼容 ====================

_TV_WEIGHTS_MAPPING = {
    # EfficientNet 系列
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "efficientnet_b1": "EfficientNet_B1_Weights",
    "efficientnet_b2": "EfficientNet_B2_Weights",
    "efficientnet_b3": "EfficientNet_B3_Weights",
    "efficientnet_b4": "EfficientNet_B4_Weights",
    "efficientnet_b5": "EfficientNet_B5_Weights",
    "efficientnet_b6": "EfficientNet_B6_Weights",
    "efficientnet_b7": "EfficientNet_B7_Weights",
    "efficientnet_v2_s": "EfficientNet_V2_S_Weights",
    "efficientnet_v2_m": "EfficientNet_V2_M_Weights",
    "efficientnet_v2_l": "EfficientNet_V2_L_Weights",
    # MobileNet 系列
    "mobilenet_v2": "MobileNet_V2_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    # ShuffleNet 系列
    "shufflenet_v2_x0_5": "ShuffleNet_V2_X0_5_Weights",
    "shufflenet_v2_x1_0": "ShuffleNet_V2_X1_0_Weights",
    "shufflenet_v2_x1_5": "ShuffleNet_V2_X1_5_Weights",
    "shufflenet_v2_x2_0": "ShuffleNet_V2_X2_0_Weights",
    # SqueezeNet 系列
    "squeezenet1_0": "SqueezeNet1_0_Weights",
    "squeezenet1_1": "SqueezeNet1_1_Weights",
    # MNASNet 系列
    "mnasnet0_5": "MNASNet0_5_Weights",
    "mnasnet0_75": "MNASNet0_75_Weights",
    "mnasnet1_0": "MNASNet1_0_Weights",
    "mnasnet1_3": "MNASNet1_3_Weights",
    # ResNet 系列（规则可推导，但明确列出更可靠）
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
    # DenseNet 系列
    "densenet121": "DenseNet121_Weights",
    "densenet161": "DenseNet161_Weights",
    "densenet169": "DenseNet169_Weights",
    "densenet201": "DenseNet201_Weights",
    # VGG 系列
    "vgg11": "VGG11_Weights",
    "vgg13": "VGG13_Weights",
    "vgg16": "VGG16_Weights",
    "vgg19": "VGG19_Weights",
    # ConvNeXt 系列
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "convnext_small": "ConvNeXt_Small_Weights",
    "convnext_base": "ConvNeXt_Base_Weights",
    "convnext_large": "ConvNeXt_Large_Weights",
    # ViT 系列
    "vit_b_16": "ViT_B_16_Weights",
    "vit_b_32": "ViT_B_32_Weights",
    "vit_l_16": "ViT_L_16_Weights",
    "vit_l_32": "ViT_L_32_Weights",
}


def _find_weights_class(model_name: str):
    """根据模型名查找对应的 torchvision Weights 类（torchvision >= 0.13）

    优先查映射表，再尝试基于命名规则推导。
    """
    if model_name in _TV_WEIGHTS_MAPPING:
        weights_name = _TV_WEIGHTS_MAPPING[model_name]
        if hasattr(models, weights_name):
            return getattr(models, weights_name).IMAGENET1K_V1

    # 兜底：通用规则推导（适用于未在映射表中的模型）
    parts = model_name.split("_")
    capitalized = "_".join(p.capitalize() for p in parts)
    candidates = [
        f"{capitalized}_Weights",
        f"{'ResNet' + model_name[6:]}_Weights" if model_name.startswith("resnet") else None,
        f"{'ResNeXt' + model_name[7:]}_Weights" if model_name.startswith("resnext") else None,
        f"Wide_ResNet{model_name[11:]}_Weights" if model_name.startswith("wide_resnet") else None,
    ]
    for c in candidates:
        if c and hasattr(models, c):
            return getattr(models, c).IMAGENET1K_V1

    raise ValueError(
        f"Cannot find weights class for {model_name!r}. "
        f"Available: {[k for k in dir(models) if k.endswith('_Weights')]}"
    )


def load_torchvision_model(model_name: str) -> nn.Module:
    """加载 torchvision 预训练模型

    自动适配 torchvision < 0.13（pretrained=True）和 >= 0.13（Weights 枚举）。
    """
    if model_name not in models.__dict__:
        raise ValueError(
            f"Model {model_name!r} not found in torchvision.models. "
            f"Available torchvision models: {[k for k in dir(models) if not k.startswith('_')]}"
        )

    model_entry = models.__dict__[model_name]
    tv_version = version.parse(models.__version__)

    if tv_version >= version.parse("0.13"):
        weights = _find_weights_class(model_name)
        print(f"[Torchvision] Loading {model_name} with weights: {weights} (v{tv_version})")
        model = model_entry(weights=weights)
    else:
        print(f"[Torchvision] Loading {model_name} with pretrained=True (v{tv_version})")
        model = model_entry(pretrained=True)

    model.eval()
    return model


# ==================== 导出器 ====================


class TorchvisionExporter(BaseExporter):
    """torchvision 分类模型 ONNX 导出器

    Args:
        model_name: torchvision.models 中的模型名，如 "efficientnet_b0", "resnet18"
        opset: ONNX opset 版本（默认 12）
    """

    def __init__(self, model_name: str, opset: int = 12):
        super().__init__(model_name, opset)
        self._model: Optional[nn.Module] = None

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = load_torchvision_model(self.model_name)
        return self._model

    def export_onnx(
        self,
        output_path: str,
        img_size: Union[int, Tuple[int, int]] = 224,
        img_channels: int = 3,
        do_validation: bool = True,
        verbose: bool = False,
    ) -> str:
        """导出 ONNX 模型

        Args:
            output_path: ONNX 保存路径
            img_size: 输入图像尺寸（int 表示正方形，tuple 为 (H, W)）
            img_channels: 输入通道数
            do_validation: 是否自动执行 ONNX 验证 + PT 输出对比
            verbose: torch.onnx.export 的 verbose 参数

        Returns:
            ONNX 文件绝对路径
        """
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        height, width = img_size
        dummy_input = torch.randn(1, img_channels, height, width)

        # 运行一次前向，确定输出结构
        model = self.model
        with torch.no_grad():
            output = model(dummy_input)

        # 统一输出命名
        if isinstance(output, (list, tuple)):
            output_names = [f"output{i}" for i in range(len(output))]
        else:
            output_names = ["output0"]
            output = [output]

        print(f"[Torchvision] Exporting {self.model_name} -> {output_path}")
        print(f"[Torchvision]   Input shape: (1, {img_channels}, {height}, {width})")
        print(f"[Torchvision]   Outputs: {output_names}")

        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=output_path,
            export_params=True,
            opset_version=self.opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=output_names,
            dynamic_axes=None,
            verbose=verbose,
        )

        print(f"[Torchvision] ✅ ONNX saved to {output_path}")

        # 自动验证
        if do_validation:
            check_onnx(output_path)
            compare_output(
                onnx_path=output_path,
                input_tensor=dummy_input,
                torch_model=model,
                torch_outputs=output,
            )

        return output_path


# ==================== CLI ====================


def parse_opt():
    parser = argparse.ArgumentParser(description="torchvision 分类模型 ONNX 导出")
    parser.add_argument(
        "--model", type=str, required=True,
        help="torchvision 模型名，如 efficientnet_b0, resnet18",
    )
    parser.add_argument(
        "--save", type=str, required=True,
        help="ONNX 保存路径",
    )
    parser.add_argument(
        "--img-size", type=int, nargs="+", default=[224],
        help="输入尺寸（单个值或 H W）",
    )
    parser.add_argument(
        "--img-channels", type=int, default=3,
    )
    parser.add_argument(
        "--opset", type=int, default=12,
    )
    return parser.parse_args()


def main():
    args = parse_opt()
    if len(args.img_size) == 1:
        img_size = args.img_size[0]
    else:
        img_size = tuple(args.img_size[:2])

    exporter = TorchvisionExporter(args.model, opset=args.opset)
    exporter.export_onnx(
        output_path=args.save,
        img_size=img_size,
        img_channels=args.img_channels,
    )


if __name__ == "__main__":
    main()
