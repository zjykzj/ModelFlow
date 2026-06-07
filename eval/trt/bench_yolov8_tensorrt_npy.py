# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : bench_yolov8_tensorrt_npy.py
@Author  : zj
@Description: YOLOv8 TensorRT 检测评估（使用 modelflow）

用法:
    python3 eval/trt/bench_yolov8_tensorrt_npy.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modelflow.pipelines import create_detect_pipeline
from modelflow.datasets import COCODetectionDataset
from modelflow.evaluators import DetectEvaluator
from modelflow.cfgs.coco import class_list
from modelflow.utils import Profile, get_logger

logger = get_logger("bench.yolov8.trt")

if __name__ == '__main__':
    with Profile(name="evaluating"):
        model_path = "./models/tensorrt/yolov8s_fp16.engine"
        data_root = "/home/zjykzj/datasets/coco/"

        pipeline = create_detect_pipeline(
            model_path=model_path, class_list=class_list, backend="tensorrt",
        )
        dataset = COCODetectionDataset(
            os.path.join(data_root, "val2017"), class_list,
            anno_json=os.path.join(data_root, "annotations/instances_val2017.json"),
        )
        evaluator = DetectEvaluator(pipeline, dataset, gt_json=dataset.get_gt_json())
        results = evaluator.run()
        print(results)
