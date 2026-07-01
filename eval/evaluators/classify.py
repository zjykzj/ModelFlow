# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : classify.py
@Author  : zj
@Description: 分类评估器

使用本地 ClassificationMetrics（DataFlow-CV 暂无分类指标）。

用法:
    evaluator = ClassifyEvaluator(pipeline, dataset, num_classes=1000)
    results = evaluator.run()
    # {"accuracy": 0.85, "f1_macro": 0.83, ...}
"""

from typing import Dict

from tqdm import tqdm

from eval.interfaces import BaseEvaluator
from eval.metrics.classification import ClassificationMetrics
from utils.logger import get_logger

logger = get_logger("eval.evaluator.classify")


class ClassifyEvaluator(BaseEvaluator):
    """分类评估器

    Args:
        pipeline: InferencePipeline 实例
        dataset: 数据集实例
        metrics: ClassificationMetrics 实例（不传则自动创建）
        config: 评估配置
    """

    def __init__(self, pipeline, dataset, metrics=None, config=None):
        if metrics is None:
            num_classes = getattr(dataset, "num_classes",
                                  len(getattr(dataset, "class_list", [])))
            metrics = ClassificationMetrics(num_classes=num_classes)
        super().__init__(pipeline, dataset, metrics, config)

    def run(self) -> Dict[str, float]:
        """运行评估"""
        self.metrics.reset()

        for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
            image, gt = self.dataset[idx]
            result = self.pipeline(image)
            self.metrics.update(result, gt)

        return self.metrics.compute()
