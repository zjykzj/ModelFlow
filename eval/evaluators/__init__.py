# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : __init__.py
@Author  : zj
@Description: 评估器

评估器负责编排：Pipeline + Dataset + Metrics。
检测/分割评估委托 DataFlow-CV 计算 mAP。
分类评估使用本地 ClassificationMetrics。
"""

from .base import BaseEvaluator
from .detect import DetectEvaluator
from .classify import ClassifyEvaluator
from .segment import SegmentEvaluator

__all__ = ["BaseEvaluator", "DetectEvaluator", "ClassifyEvaluator", "SegmentEvaluator"]
