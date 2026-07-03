# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : preprocess.py
@Author  : zj
@Description: 语义分割预处理 — resize + normalize

与分类预处理类似，但不做 center crop（保持原图比例）。
直接缩放至目标尺寸或保持原图尺寸。

流程: OpenCV BGR → RGB → Resize(H, W) → Normalize → CHW
"""

from typing import Tuple, Union

import cv2
import numpy as np

from modelflow.interfaces import BasePreprocessor


class SemanticSegPreprocessor(BasePreprocessor):
    """语义分割预处理（NumPy 实现）

    Args:
        target_size: 目标尺寸 (H, W)，None 表示保持原图尺寸
        mean: 均值（数据集统计值，默认 ImageNet）
        std: 标准差（默认 ImageNet）
    """

    def __init__(
        self,
        target_size: Union[Tuple[int, int], None] = None,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if image.size == 0:
            raise ValueError("Empty input image")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.target_size is not None:
            h, w = self.target_size
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std

        chw = np.transpose(normalized, (2, 0, 1))
        return chw[None]  # (1, 3, H, W)
