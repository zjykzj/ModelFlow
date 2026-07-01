# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : test_eval.py
@Author  : zj
@Description: eval/ 模块单元测试

覆盖: BaseEvaluator, BaseMetrics
"""

import pytest

from eval.interfaces import BaseEvaluator, BaseMetrics


class TestEvalInterfaces:
    """eval ABC 接口测试"""

    def test_base_metrics_interface(self):
        """BaseMetrics ABC 接口合规"""
        class MyMetrics(BaseMetrics):
            def update(self, p, g): pass
            def compute(self): return {"acc": 0.5}
            def reset(self): pass
        m = MyMetrics()
        assert m.compute()["acc"] == 0.5

    def test_base_evaluator_interface(self):
        """BaseEvaluator ABC 接口合规"""
        class MyEval(BaseEvaluator):
            def run(self): return {"mAP": 0.5}
        ev = MyEval(None, None)
        assert ev.run()["mAP"] == 0.5

    def test_base_evaluator_ctor(self):
        """BaseEvaluator 构造函数保存参数"""
        pipeline = object()
        dataset = object()
        metrics = object()
        config = {"key": "value"}

        class MyEval(BaseEvaluator):
            def run(self): return {}

        ev = MyEval(pipeline, dataset, metrics=metrics, config=config)
        assert ev.pipeline is pipeline
        assert ev.dataset is dataset
        assert ev.metrics is metrics
        assert ev.config is config

    def test_base_evaluator_default_config(self):
        """BaseEvaluator config 默认为空 dict"""
        class MyEval(BaseEvaluator):
            def run(self): return {}

        ev = MyEval(None, None)
        assert ev.config == {}
