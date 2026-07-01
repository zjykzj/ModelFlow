# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : base.py
@Author  : zj
@Description: BaseDataset — 数据集抽象基类

所有数据集必须实现 __len__, __getitem__, get_gt_json 三个方法。
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class BaseDataset(ABC):
    """数据集基类

    契约:
        - __len__: 返回样本总数
        - __getitem__: 返回 (image: HWC BGR ndarray, ground_truth: dict)
        - get_gt_json: 返回 ground truth JSON 路径（用于评估），无则返回 ""
    """

    @abstractmethod
    def __len__(self) -> int:
        """返回数据集大小"""
        ...

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[np.ndarray, Dict]:
        """获取样本

        Returns:
            (image: HWC BGR ndarray, ground_truth: dict)
        """
        ...

    @abstractmethod
    def get_gt_json(self) -> str:
        """获取 ground truth JSON 文件路径

        Returns:
            JSON 文件路径，无标注则返回空字符串
        """
        ...
