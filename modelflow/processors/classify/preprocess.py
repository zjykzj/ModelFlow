# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : preprocess.py
@Author  : zj
@Description: 分类预处理 — ImageNet 标准预处理 (NumPy 实现)

流程: OpenCV BGR → RGB → Resize(256) → CenterCrop(224) → Normalize → CHW
"""

from typing import Union, Tuple

import cv2
import numpy as np

from modelflow.core.interfaces import BasePreprocessor
from modelflow.core.registry import PROCESSORS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@PROCESSORS.register("classify_preprocess_npy")
class ClassifyPreprocessor(BasePreprocessor):
    """分类预处理（NumPy 实现）

    Args:
        crop_size: 裁剪尺寸（默认 224）
        resize_size: 缩放短边尺寸（默认 256）
    """

    def __init__(self, crop_size: int = 224, resize_size: int = 256):
        self.crop_size = crop_size
        self.resize_size = resize_size

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]

        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize 短边
        new_h, new_w = (self.resize_size, int(w * self.resize_size / h)) if h < w else \
                       (int(h * self.resize_size / w), self.resize_size)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # CenterCrop
        ch, cw = resized.shape[:2]
        sh = (ch - self.crop_size) // 2
        sw = (cw - self.crop_size) // 2
        cropped = resized[sh:sh + self.crop_size, sw:sw + self.crop_size]

        # Normalize
        normalized = cropped.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

        # HWC → CHW → NCHW
        chw = np.transpose(normalized, (2, 0, 1))
        return chw[None]  # 添加 batch 维度
