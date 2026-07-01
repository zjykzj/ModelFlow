# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_metrics.py
@Author  : zj
@Description: 指标模块测试

覆盖: ClassificationMetrics
"""

import pytest
import numpy as np

from eval.metrics.classification import ClassificationMetrics


class TestClassificationMetrics:
    """分类指标测试"""

    def test_initial_state(self):
        m = ClassificationMetrics(num_classes=3)
        result = m.compute()
        assert result["accuracy"] == 0.0  # 无数据时 accuracy 为 0
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "f1_macro" in result

    def test_perfect_classification(self):
        m = ClassificationMetrics(num_classes=3)
        for i in range(3):
            m.update({"class_ids": [i]}, {"class_id": i})
        result = m.compute()
        assert result["accuracy"] == pytest.approx(1.0, rel=1e-6)
        assert result["f1_macro"] == pytest.approx(1.0, rel=1e-6)

    def test_all_wrong(self):
        m = ClassificationMetrics(num_classes=3)
        for i in range(3):
            wrong = (i + 1) % 3
            m.update({"class_ids": [wrong]}, {"class_id": i})
        result = m.compute()
        assert result["accuracy"] == 0.0

    def test_reset(self):
        m = ClassificationMetrics(num_classes=3)
        m.update({"class_ids": [0]}, {"class_id": 0})
        m.reset()
        result = m.compute()
        assert result["accuracy"] == 0.0

    def test_multi_class_metrics(self):
        m = ClassificationMetrics(num_classes=5)
        # 3 正确，2 错误
        for i in range(3):
            m.update({"class_ids": [i]}, {"class_id": i})
        m.update({"class_ids": [1]}, {"class_id": 0})
        m.update({"class_ids": [2]}, {"class_id": 3})
        result = m.compute()
        assert 0.5 < result["accuracy"] < 0.7  # 3/5

    def test_reset_after_update(self):
        m = ClassificationMetrics(num_classes=2)
        m.update({"class_ids": [0]}, {"class_id": 0})
        m.update({"class_ids": [1]}, {"class_id": 1})
        assert m.compute()["accuracy"] == pytest.approx(1.0)
        m.reset()
        m.update({"class_ids": [0]}, {"class_id": 0})
        assert m.compute()["accuracy"] == pytest.approx(1.0)
