# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : __init__.py
@Author  : zj
@Description: ModelFlow — Python 推理/评估/可视化核心包
"""

__version__ = "0.6.0"

from .types import ModelInfo
from .interfaces import (
    BaseBackend,
    BasePreprocessor,
    BasePostprocessor,
    InferencePipeline,
)

__all__ = [
    "ModelInfo",
    "BaseBackend", "BasePreprocessor", "BasePostprocessor",
    "InferencePipeline",
]
