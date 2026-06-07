# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : bench_classify_tensorrt_npy.py
@Author  : zj
@Description: 分类 TensorRT 评估（使用 modelflow）

用法:
    python3 eval/trt/bench_classify_tensorrt_npy.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modelflow.pipelines import create_classify_pipeline
from modelflow.datasets import ClassifyDataset
from modelflow.evaluators import ClassifyEvaluator
from modelflow.utils import Profile, get_logger
from modelflow.cfgs.imagenet import get_imagenet_classes

logger = get_logger("bench.classify.trt")

if __name__ == '__main__':
    with Profile(name="evaluating"):
        model_path = "./models/tensorrt/efficientnet_b0_fp16.engine"
        data_root = "/home/zjykzj/datasets/imagenet/val"
        class_list = get_imagenet_classes()

        pipeline = create_classify_pipeline(
            model_path=model_path, class_list=class_list, backend="tensorrt",
        )
        dataset = ClassifyDataset(data_root, class_list)
        evaluator = ClassifyEvaluator(pipeline, dataset)
        results = evaluator.run()
        print(results)
