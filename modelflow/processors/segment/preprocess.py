# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : preprocess.py
@Author  : zj
@Description: 分割预处理 — 同检测预处理（LetterBox）
"""

from modelflow.core.registry import PROCESSORS
from modelflow.processors.detect.preprocess import DetectPreprocessor


@PROCESSORS.register("segment_preprocess_npy")
class SegmentPreprocessor(DetectPreprocessor):
    """分割预处理 — 与检测预处理相同（LetterBox）

    实例分割模型（如 YOLOv8-seg）使用与检测模型相同的预处理流程。
    """
    pass
