# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : types.py
@Author  : zj
@Description: 枚举类型和数据结构
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


class TaskType(str, Enum):
    """视觉任务类型"""
    CLASSIFY = "classify"
    DETECT = "detect"
    INSTANCE_SEGMENT = "instance_segment"
    SEMANTIC_SEGMENT = "semantic_segment"
    MULTIMODAL = "multimodal"
    POSE = "pose"


class BackendType(str, Enum):
    """推理后端类型"""
    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"
    TRITON = "triton"


class ProcessorType(str, Enum):
    """预处理/后处理实现类型"""
    NUMPY = "numpy"
    TORCH = "torch"


@dataclass
class ModelInfo:
    """模型输入/输出的元信息"""
    name: str
    shape: List[int]
    dtype: np.dtype
    index: int = 0
