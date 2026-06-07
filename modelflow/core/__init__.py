# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: modelflow core — 基础设施
"""

from .types import TaskType, BackendType, ProcessorType, ModelInfo
from .registry import Registry, BACKENDS, PROCESSORS, DATASETS, METRICS, EVALUATORS
from .interfaces import (
    BaseBackend,
    BasePreprocessor,
    BasePostprocessor,
    BaseDataset,
    BaseMetrics,
    BaseEvaluator,
    BaseVisualizer,
    InferencePipeline,
)

__all__ = [
    "TaskType", "BackendType", "ProcessorType", "ModelInfo",
    "Registry", "BACKENDS", "PROCESSORS", "DATASETS", "METRICS", "EVALUATORS",
    "BaseBackend", "BasePreprocessor", "BasePostprocessor",
    "BaseDataset", "BaseMetrics", "BaseEvaluator", "BaseVisualizer",
    "InferencePipeline",
]
