# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 15:08
@File    : utils.py
@Author  : zj
@Description: 

Utility functions for model loading, input shape inference, and output path handling.

This module provides:
- Model loading from various sources (torchvision, local file, toy)
- Input shape estimation based on model source
- Output path preparation with directory creation

"""

import os
import onnx

import numpy as np


def get_output_path(save_path: str) -> str:
    """
    Ensure the output directory exists and return the normalized save path.

    Args:
        save_path (str): Desired output path.

    Returns:
        str: Normalized and absolute output path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return os.path.abspath(save_path)


def load_dummy_input_from_onnx(onnx_path: str, batch_size: int = 1) -> np.ndarray:
    """
    Load input shape info from ONNX model and generate dummy input.

    Returns:
        np.ndarray: Dummy input with shape [B, C, H, W]
    """
    model = onnx.load(onnx_path)
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    if input_shape[0] == 0:
        input_shape[0] = batch_size  # dynamic batch -> fix it
    dummy = np.random.rand(*input_shape).astype(np.float32)
    return dummy
