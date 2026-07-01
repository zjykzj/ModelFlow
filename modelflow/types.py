# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : types.py
@Author  : zj
@Description: 数据结构
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ModelInfo:
    """模型输入/输出的元信息"""
    name: str
    shape: List[int]
    dtype: np.dtype
    index: int = 0
