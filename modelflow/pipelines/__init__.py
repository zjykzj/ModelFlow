# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: 预构建 Pipeline 工厂
"""

from .classify import create_classify_pipeline
from .detect import create_detect_pipeline
from .segment import create_segment_pipeline

__all__ = ["create_classify_pipeline", "create_detect_pipeline", "create_segment_pipeline"]
