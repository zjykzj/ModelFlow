# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : convert.py
@Author  : zj
@Description: Torchvision classification model ONNX exporter

Supports ResNet, EfficientNet, MobileNet, ShuffleNet, SqueezeNet,
MNASNet, DenseNet, VGG, ConvNeXt, ViT, and other mainstream classification models.

Typical usage:
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

from export._base import BaseExporter
from export._validation import check_onnx, compare_output


# ==================== Torchvision Version Compatibility ====================

_TV_WEIGHTS_MAPPING = {
    # EfficientNet family
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
    # MobileNet family
    "mobilenet_v2": "MobileNet_V2_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    # ShuffleNet family
    "shufflenet_v2_x0_5": "ShuffleNet_V2_X0_5_Weights",
    "shufflenet_v2_x1_0": "ShuffleNet_V2_X1_0_Weights",
    "shufflenet_v2_x1_5": "ShuffleNet_V2_X1_5_Weights",
    "shufflenet_v2_x2_0": "ShuffleNet_V2_X2_0_Weights",
    # SqueezeNet family
    "squeezenet1_0": "SqueezeNet1_0_Weights",
    "squeezenet1_1": "SqueezeNet1_1_Weights",
    # MNASNet family
    "mnasnet0_5": "MNASNet0_5_Weights",
    "mnasnet0_75": "MNASNet0_75_Weights",
    "mnasnet1_0": "MNASNet1_0_Weights",
    "mnasnet1_3": "MNASNet1_3_Weights",
    # ResNet family (derivable by rule, but explicit listing is more reliable)
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
    # DenseNet family
    "densenet121": "DenseNet121_Weights",
    "densenet161": "DenseNet161_Weights",
    "densenet169": "DenseNet169_Weights",
    "densenet201": "DenseNet201_Weights",
    # VGG family
    "vgg11": "VGG11_Weights",
    "vgg13": "VGG13_Weights",
    "vgg16": "VGG16_Weights",
    "vgg19": "VGG19_Weights",
    # ConvNeXt family
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "convnext_small": "ConvNeXt_Small_Weights",
    "convnext_base": "ConvNeXt_Base_Weights",
    "convnext_large": "ConvNeXt_Large_Weights",
    # ViT family
    "vit_b_16": "ViT_B_16_Weights",
    "vit_b_32": "ViT_B_32_Weights",
    "vit_l_16": "ViT_L_16_Weights",
    "vit_l_32": "ViT_L_32_Weights",
}


def _find_weights_class(model_name: str):
    """Look up the corresponding torchvision Weights class by model name (torchvision >= 0.13)

    First checks the mapping table, then falls back to naming-convention derivation.
    """
    if model_name in _TV_WEIGHTS_MAPPING:
        weights_name = _TV_WEIGHTS_MAPPING[model_name]
        if hasattr(models, weights_name):
            return getattr(models, weights_name).IMAGENET1K_V1

    # Fallback: generic rule-based derivation (for models not in the mapping table)
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
    """Load a torchvision pretrained model

    Automatically adapts to torchvision < 0.13 (pretrained=True) and >= 0.13 (Weights enum).
    """
    if model_name not in models.__dict__:
        raise ValueError(
            f"Model {model_name!r} not found in torchvision.models. "
            f"Available torchvision models: {[k for k in dir(models) if not k.startswith('_')]}"
        )

    model_entry = models.__dict__[model_name]
    import torchvision
    tv_version = version.parse(torchvision.__version__)

    if tv_version >= version.parse("0.13"):
        weights = _find_weights_class(model_name)
        print(f"[Torchvision] Loading {model_name} with weights: {weights} (v{tv_version})")
        model = model_entry(weights=weights)
    else:
        print(f"[Torchvision] Loading {model_name} with pretrained=True (v{tv_version})")
        model = model_entry(pretrained=True)

    model.eval()
    return model


# ==================== Exporter ====================


class TorchvisionExporter(BaseExporter):
    """Torchvision classification model ONNX exporter

    Args:
        model_name: Model name in torchvision.models, e.g. "efficientnet_b0", "resnet18"
        opset: ONNX opset version (default 12)
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
        """Export an ONNX model

        Args:
            output_path: ONNX save path
            img_size: Input image size (int for square, tuple for (H, W))
            img_channels: Number of input channels
            do_validation: Whether to automatically run ONNX validation and PT output comparison
            verbose: torch.onnx.export verbose parameter

        Returns:
            Absolute path to the ONNX file
        """
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        height, width = img_size
        dummy_input = torch.randn(1, img_channels, height, width)

        # Run one forward pass to determine output structure
        model = self.model
        with torch.no_grad():
            output = model(dummy_input)

        # Unify output naming
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

        # Auto-validation
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
    parser = argparse.ArgumentParser(description="Torchvision classification model ONNX export")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Torchvision model name, e.g. efficientnet_b0, resnet18",
    )
    parser.add_argument(
        "--save", type=str, required=True,
        help="ONNX save path",
    )
    parser.add_argument(
        "--img-size", type=int, nargs="+", default=[224],
        help="Input size (single value or H W)",
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
