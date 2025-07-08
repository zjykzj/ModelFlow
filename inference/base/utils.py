# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:03
@File    : util.py
@Author  : zj
@Description: 工具函数（图像尺寸转换等）
"""

import os
import numpy as np

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")


def load_class_names(class_name_file: str) -> list:
    """
    从 .names 文件加载类别名称列表

    :param class_name_file: 文件名（如 'coco.names'），或完整路径
    :return: 类别名称列表
    """
    if not class_name_file.endswith(".names"):
        raise ValueError("Only support loading '.names' files")

    file_path = os.path.join(ASSETS_DIR, class_name_file)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Class name file not found: {file_path}")

    with open(file_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]

    return class_names


def softmax(x: list or np.ndarray) -> np.ndarray:
    """
    Softmax 函数实现
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)
