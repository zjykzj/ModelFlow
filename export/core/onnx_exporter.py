# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 13:56
@File    : base_exporter.py
@Author  : zj
@Description: 

Module for exporting PyTorch models to ONNX format.

This module provides the main function `export_to_onnx` which handles:
- Model loading from various sources (torchvision, local file, toy)
- Dummy input generation based on input shape
- ONNX export with optional dynamic batch support
- ONNX model verification

"""

from typing import Optional, List

import torch
import torchvision

from core.onnx_verify import print_onnx_model_info, validate_onnx_model, verify_torch_onnx

# Supported torchvision classification models (manually curated)
SUPPORTED_MODELS = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
    'toy'  # Toy classifier for testing
]


def get_torchvision_model_names() -> List[str]:
    """
    Retrieve a list of available torchvision classification model names.

    Returns:
        List[str]: Sorted list of valid torchvision model names.
    """
    model_names = sorted(
        name for name in dir(torchvision.models)
        if name.islower()
        and not name.startswith("__")
        and callable(getattr(torchvision.models, name))
    )
    return model_names


def load_model(
        model_name: str,
        pretrain_path: Optional[str] = None,
        num_classes: int = 1000,
) -> torch.nn.Module:
    """
    Load a PyTorch classification model from torchvision or local file.

    Args:
        model_name (str): Name of the model (e.g., 'resnet50', 'mobilenet_v2', 'toy').
        pretrain_path (Optional[str]): Path to local .pt/.pth file containing state_dict.
        num_classes (int): Number of output classes for final classification layer.

    Returns:
        torch.nn.Module: A loaded and optionally modified PyTorch model in eval mode.

    Raises:
        ValueError: If model name is not supported.
        RuntimeError: If loading weights fails.
    """
    # Step 1: Toy classifier for testing
    if model_name == "toy":
        print("Using ToyClassifier for testing")
        from models.toy_net import ToyNet
        model = ToyNet(num_classes=num_classes)

    # Step 2: Load model from torchvision
    elif model_name in SUPPORTED_MODELS:
        print(f"Creating model '{model_name}' from torchvision")
        try:
            model = torchvision.models.__dict__[model_name](pretrained=True)

            # Modify final classification layer if needed
            if num_classes != 1000:

                # 🔁 ResNet 系列: 使用 'fc' 层作为最终输出
                if hasattr(model, "fc"):
                    in_features = model.fc.in_features
                    model.fc = torch.nn.Linear(in_features, num_classes)
                    print(f"Replaced ResNet-style 'fc' layer with {num_classes} outputs")

                # 📱 MobileNet / ShuffleNet / EfficientNet: 使用 'classifier' 层
                elif hasattr(model, "classifier"):

                    # 🔍 EfficientNet: classifier 是一个 Sequential([Dropout, Linear]))
                    if isinstance(model.classifier, torch.nn.Sequential):
                        last_module = model.classifier[-1]
                        in_features = last_module.in_features
                        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
                        print(f"Replaced EfficientNet-style 'classifier[-1]' layer with {num_classes} outputs")

                    # 🚶‍♂️ MobileNet/ShuffleNet: classifier 是单一 Linear 层
                    else:
                        in_features = model.classifier.in_features
                        model.classifier = torch.nn.Linear(in_features, num_classes)
                        print(f"Replaced MobileNet/ShuffleNet-style 'classifier' layer with {num_classes} outputs")

                else:
                    raise RuntimeError(f"Unsupported model architecture: {model_name}. "
                                       f"No known classification head ('fc' or 'classifier') found.")

        except Exception as e:
            raise RuntimeError(f"Failed to create model '{model_name}': {e}")

    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. "
            f"Supported models: {SUPPORTED_MODELS} + 'toy'"
        )

    # Step 3: Load custom pretrained weights if provided
    if pretrain_path and os.path.isfile(pretrain_path):
        print(f"Loading custom weights from: {pretrain_path}")
        state_dict = torch.load(pretrain_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.eval()
    return model


def export_to_onnx(
        model_name: str,
        save_path: str,
        pretrain_path: Optional[str] = None,
        num_classes: int = 1000,
        input_shape: Optional[List[int]] = None,
        dynamic_axes: bool = False,
        opset: int = 12,
):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model_name (str): Name of the model (e.g., 'resnet50'), or 'toy' for test model.
        save_path (str): Path to save the exported ONNX model.
        pretrain_path (Optional[str]): Path to local pre-trained .pt/.pth file.
        num_classes (int): Number of output classes. Default: 1000 (ImageNet).
        input_shape (Optional[List[int]]): Input shape as [C, H, W]. Default: [3, 224, 224].
        dynamic_axes (bool): Whether to enable dynamic batch size in ONNX.
        opset (int): ONNX opset version. Default: 12.
    """
    # Load the model based on provided configuration
    model = load_model(
        model_name=model_name,
        pretrain_path=pretrain_path,
        num_classes=num_classes
    )

    # 默认输入 shape 设置
    if input_shape is None:
        if 'toy' in model_name:
            input_shape = [1, 32, 32]  # MNIST-like (grayscale)
        else:
            input_shape = [3, 224, 224]  # ImageNet-like (RGB)

    # 确保 input_shape 是一个 list/tuple 且长度为 3
    assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 3, \
        f"Input shape must be a list/tuple of length 3 (C, H, W), got {input_shape}"

    # 添加 batch 维度
    dummy_input = torch.randn(1, *input_shape)

    # Set dynamic axes configuration
    dynamic_axes_config = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    } if dynamic_axes else None

    print(f"Exporting model '{model_name}' to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,  # Store trained parameter weights inside the model file
        opset_version=opset,  # ONNX opset version
        do_constant_folding=True,  # Optimize constants
        input_names=['input'],  # Input tensor name
        output_names=['output'],  # Output tensor name
        dynamic_axes=dynamic_axes_config
    )

    print(f"✅ Successfully exported ONNX model to: {save_path}")

    # Print ONNX model info
    print_onnx_model_info(save_path)

    # Validate ONNX model
    validate_onnx_model(save_path)

    # Verify numerical consistency
    verify_torch_onnx(model, dummy_input, save_path)
