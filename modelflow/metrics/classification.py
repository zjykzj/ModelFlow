# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : classification.py
@Author  : zj
@Description: 分类指标 — 混淆矩阵 → Accuracy / Precision / Recall / F1

DataFlow-CV 暂无分类 metric，此处本地实现。
"""

import numpy as np
from typing import Dict

from modelflow.core.interfaces import BaseMetrics
from modelflow.core.registry import METRICS


@METRICS.register("classification")
class ClassificationMetrics(BaseMetrics):
    """分类指标

    基于混淆矩阵计算 Accuracy / Precision / Recall / F1。

    Args:
        num_classes: 类别数
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def update(self, prediction, ground_truth):
        """累加一个样本

        Args:
            prediction: dict 包含 class_ids（预测类别 ID 列表）
            ground_truth: dict 包含 class_id（真实类别 ID）
        """
        pred_class = prediction["class_ids"][0] if isinstance(prediction.get("class_ids"), list) else prediction.get("class_ids")
        if isinstance(pred_class, (list, np.ndarray)):
            pred_class = pred_class[0]
        gt_class = ground_truth["class_id"]
        self.confusion_matrix[gt_class, pred_class] += 1

    def compute(self) -> Dict[str, float]:
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        accuracy = tp.sum() / (self.confusion_matrix.sum() + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()),
            "f1_macro": float(f1.mean()),
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
