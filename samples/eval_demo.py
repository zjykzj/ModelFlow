# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : eval_demo.py
@Author  : zj
@Description: 评估工作流示例（使用 modelflow）

演示如何使用 modelflow 搭建完整的评估流程。

用法:
    python3 samples/eval_demo.py --task detect --model yolov8s.onnx --data /path/to/coco
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelflow.utils import Profile, get_logger

logger = get_logger("samples.eval_demo")


def parse_opt():
    parser = argparse.ArgumentParser(description="ModelFlow 评估工作流示例")
    parser.add_argument("--task", type=str, default="detect",
                        choices=["classify", "detect", "segment"],
                        help="任务类型")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--data", type=str, required=True, help="数据集根目录")
    parser.add_argument("--backend", type=str, default="onnxruntime",
                        choices=["onnxruntime", "tensorrt", "triton"],
                        help="推理后端")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU 阈值")
    parser.add_argument("--save-pred", type=str, default=None, help="预测结果保存路径")
    parser.add_argument("--model-version", type=str, default="v8",
                        choices=["v5", "v8", "v11"], help="YOLO 版本")
    return parser.parse_args()


def main():
    args = parse_opt()
    data_root = args.data

    with Profile("eval_demo") as perf:
        if args.task == "classify":
            from modelflow.pipelines import create_classify_pipeline
            from modelflow.datasets import ClassifyDataset
            from modelflow.evaluators import ClassifyEvaluator
            from modelflow.cfgs.imagenet import get_imagenet_classes

            class_list = get_imagenet_classes()
            pipeline = create_classify_pipeline(
                model_path=args.model, class_list=class_list,
                backend=args.backend,
            )
            dataset = ClassifyDataset(os.path.join(data_root, "val"), class_list)
            evaluator = ClassifyEvaluator(pipeline, dataset)
            results = evaluator.run()
            logger.info(f"Classification results: {results}")

        elif args.task == "detect":
            from modelflow.pipelines import create_detect_pipeline
            from modelflow.datasets import COCODetectionDataset
            from modelflow.evaluators import DetectEvaluator
            from modelflow.cfgs.coco import class_list

            pipeline = create_detect_pipeline(
                model_path=args.model, class_list=class_list,
                backend=args.backend, model_version=args.model_version,
                conf_thres=args.conf, iou_thres=args.iou,
            )
            dataset = COCODetectionDataset(
                os.path.join(data_root, "val2017"), class_list,
                anno_json=os.path.join(data_root, "annotations/instances_val2017.json"),
            )
            evaluator = DetectEvaluator(pipeline, dataset, gt_json=dataset.get_gt_json())
            results = evaluator.run(save_pred_json=args.save_pred)
            logger.info(f"Detection results: {results}")

        elif args.task == "segment":
            from modelflow.pipelines import create_segment_pipeline
            from modelflow.datasets import COCOSegmentDataset
            from modelflow.evaluators import SegmentEvaluator
            from modelflow.cfgs.coco import class_list

            pipeline = create_segment_pipeline(
                model_path=args.model, class_list=class_list,
                backend=args.backend, conf_thres=args.conf, iou_thres=args.iou,
            )
            dataset = COCOSegmentDataset(
                os.path.join(data_root, "val2017"), class_list,
                anno_json=os.path.join(data_root, "annotations/instances_val2017.json"),
            )
            evaluator = SegmentEvaluator(pipeline, dataset, gt_json=dataset.get_gt_json())
            results = evaluator.run(save_pred_json=args.save_pred)
            logger.info(f"Segmentation results: {results}")

    print(f"\n⏱ Total time: {perf.elapsed * 1000:.1f} ms")


if __name__ == "__main__":
    main()
