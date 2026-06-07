# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : preprocess.py
@Author  : zj
@Description: 检测预处理 — YOLO 标准 LetterBox

支持 v5/v8/v11 版本差异（auto pad 参数不同）。

流程: LetterBox → BGR→RGB → /255 → HWC→CHW
"""

from typing import Tuple, Union

import cv2
import numpy as np

from modelflow.core.interfaces import BasePreprocessor
from modelflow.core.registry import PROCESSORS


def letterbox(
    image: np.ndarray,
    target_size: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """保持宽高比的缩放 + 填充（YOLO 标准 LetterBox）

    推理模式下填充到完整的 target_size。
    YOLO 训练时的 "minimum rectangle" 模式（auto_pad=True）
    仅在训练场景使用，推理时统一填充到固定尺寸。

    Args:
        image: HWC BGR 图像
        target_size: 目标尺寸
        color: 填充颜色
        stride: 模型步长（仅用于后处理时记录缩放信息）

    Returns:
        (padded_image, scale, (pad_left, pad_top))
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color,
    )

    return padded, scale, (pad_left, pad_top)


@PROCESSORS.register("detect_preprocess_npy")
class DetectPreprocessor(BasePreprocessor):
    """检测预处理（NumPy 实现）

    所有 YOLO 版本（v5/v8/v11）的推理预处理一致：LetterBox 填充到正方形输入。
    YOLO 版本差异体现在后处理的输出格式解析中，不影响预处理。

    Args:
        input_size: 输入尺寸（默认 640）
    """

    def __init__(self, input_size: int = 640):
        self.input_size = input_size

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        padded, scale, pad = letterbox(image, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1))
        chw = chw.astype(np.float32) / 255.0
        return chw[None]  # (1, 3, H, W)
