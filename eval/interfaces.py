# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : interfaces.py
@Author  : zj
@Description: eval 模块核心接口

定义评估相关的抽象基类：
- BaseMetrics: 指标计算累加器
- BaseEvaluator: 评估编排器
"""

from abc import ABC, abstractmethod
from typing import Dict


# ==================== BaseMetrics ====================


class BaseMetrics(ABC):
    """指标计算基类（有状态累加器）

    用法:
        metrics = SomeMetrics(num_classes=80)
        for pred, gt in inference_loop:
            metrics.update(pred, gt)
        result = metrics.compute()
        metrics.reset()
    """

    @abstractmethod
    def update(self, prediction, ground_truth):
        """累加一个样本的指标"""
        ...

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """计算当前累加结果"""
        ...

    @abstractmethod
    def reset(self):
        """重置状态"""
        ...


# ==================== BaseEvaluator ====================


class BaseEvaluator(ABC):
    """评估器基类

    编排：Pipeline + Dataset + Metrics
    对检测/分割任务，metrics 委托 DataFlow-CV。
    """

    def __init__(self, pipeline, dataset, metrics=None, config=None):
        self.pipeline = pipeline
        self.dataset = dataset
        self.metrics = metrics
        self.config = config or {}

    @abstractmethod
    def run(self) -> Dict[str, float]:
        """运行完整评估流程

        Returns:
            metrics dict，如 {"mAP": 0.5, "AP50": 0.7, ...}
        """
        ...
