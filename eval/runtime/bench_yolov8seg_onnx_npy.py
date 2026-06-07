# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : bench_yolov8seg_onnx_npy.py
@Author  : zj
@Description: YOLOv8-seg ONNX Runtime 分割评估（使用 modelflow）

用法:
    python3 eval/runtime/bench_yolov8seg_onnx_npy.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modelflow.pipelines import create_segment_pipeline
from modelflow.datasets import COCOSegmentDataset
from modelflow.evaluators import SegmentEvaluator
from modelflow.cfgs.coco import class_list
from modelflow.utils import Profile, get_logger

logger = get_logger("bench.yolov8seg.onnx")

if __name__ == '__main__':
    with Profile(name="evaluating"):
        model_path = "./models/runtime/yolov8s-seg.onnx"
        data_root = "/home/zjykzj/datasets/coco/"

        pipeline = create_segment_pipeline(
            model_path=model_path, class_list=class_list, backend="onnxruntime",
        )
        dataset = COCOSegmentDataset(
            os.path.join(data_root, "val2017"), class_list,
            anno_json=os.path.join(data_root, "annotations/instances_val2017.json"),
        )
        evaluator = SegmentEvaluator(pipeline, dataset, gt_json=dataset.get_gt_json())
        results = evaluator.run()
        print(results)
