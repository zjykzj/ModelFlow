# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : __init__.py
@Author  : zj
@Description: eval 模块 — 纯评估，零 import 依赖 modelflow/ 和 data/

通过构造函数注入 pipeline 和 dataset（鸭子类型，无硬 import）：
- Evaluator 不 import modelflow 或 data，只依赖它们的接口协议
- 检测/分割评估委托 DataFlow-CV 计算 mAP
- 分类评估使用本地 ClassificationMetrics

用法:
    from modelflow.pipelines import create_detect_pipeline
    from data import build_dataset, get_class_names
    from eval import DetectEvaluator

    pipeline = create_detect_pipeline(
        model_path="yolov8s.onnx",
        class_list=get_class_names("coco"),
        backend="onnxruntime",
    )
    dataset = build_dataset("coco", path="/data/coco", val="val2017")
    evaluator = DetectEvaluator(pipeline, dataset, gt_json="annotations.json")
    results = evaluator.run()
    # {"mAP": 0.503, "AP50": 0.698, "AP75": 0.547, ...}
"""

from eval.interfaces import BaseEvaluator, BaseMetrics
from eval.evaluators import ClassifyEvaluator, DetectEvaluator, SegmentEvaluator
from eval.metrics import ClassificationMetrics

__all__ = [
    "BaseEvaluator",
    "BaseMetrics",
    "ClassifyEvaluator",
    "DetectEvaluator",
    "SegmentEvaluator",
    "ClassificationMetrics",
]
