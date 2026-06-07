# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : eval_bench.py
@Author  : zj
@Description: 统一评估基准入口（使用 modelflow）

替代 eval/runtime/, eval/triton/, eval/trt/ 下共 14 个独立 bench 脚本。

用法:
    # 检测 - ONNX Runtime
    python3 samples/eval_bench.py --task detect --model models/runtime/yolov8s.onnx --data /path/to/coco

    # 检测 - YOLOv5 ONNX Runtime
    python3 samples/eval_bench.py --task detect --model models/runtime/yolov5s.onnx --data /path/to/coco --model-version v5

    # 检测 - TensorRT
    python3 samples/eval_bench.py --task detect --model models/tensorrt/yolov8s_fp16.engine --backend tensorrt --data /path/to/coco

    # 检测 - Triton
    python3 samples/eval_bench.py --task detect --model Detect_COCO_YOLOv8s_ONNX --backend triton --data /path/to/coco --server-url localhost:8001

    # 分类 - ONNX Runtime
    python3 samples/eval_bench.py --task classify --model models/runtime/efficientnet_b0.onnx --data /path/to/imagenet --input-size 224

    # 分类 - Triton
    python3 samples/eval_bench.py --task classify --model Classify_ImageNet_EfficientNetB0_ONNX --backend triton --data /path/to/imagenet --input-size 224

    # 分割 - ONNX Runtime
    python3 samples/eval_bench.py --task segment --model models/runtime/yolov8s-seg.onnx --data /path/to/coco

    # 分割 - Triton
    python3 samples/eval_bench.py --task segment --model Segment_COCO_YOLOv8sSeg_ONNX --backend triton --data /path/to/coco
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelflow.utils import Profile, get_logger

logger = get_logger("samples.eval_bench")


def parse_opt():
    parser = argparse.ArgumentParser(description="ModelFlow 统一评估基准")
    parser.add_argument("--task", type=str, required=True,
                        choices=["classify", "detect", "segment"],
                        help="任务类型")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 Triton 模型名")
    parser.add_argument("--data", type=str, required=True,
                        help="数据集根目录")
    parser.add_argument("--backend", type=str, default="onnxruntime",
                        choices=["onnxruntime", "tensorrt", "triton"],
                        help="推理后端（默认 onnxruntime）")
    parser.add_argument("--model-version", type=str, default="v8",
                        choices=["v5", "v8", "v11"],
                        help="YOLO 版本（仅 detect 任务，默认 v8）")
    parser.add_argument("--input-size", type=int, default=640,
                        help="输入尺寸（分类默认 224，检测/分割默认 640）")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="置信度阈值（评估通常用 0.001）")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU 阈值（评估通常用 0.7）")
    parser.add_argument("--server-url", type=str, default="localhost:8001",
                        help="Triton Server 地址（仅 triton 后端）")
    parser.add_argument("--save-pred", type=str, default=None,
                        help="预测结果 JSON 保存路径（默认不保存）")
    parser.add_argument("--anno-json", type=str, default=None,
                        help="标注 JSON 路径（默认 dataset_root/annotations/instances_*.json）")
    return parser.parse_args()


def resolve_anno_json(data_root, task):
    """推断标注 JSON 路径"""
    candidates = [
        os.path.join(data_root, "annotations", "instances_val2017.json"),
        os.path.join(data_root, "annotations", "person_keypoints_val2017.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def main():
    args = parse_opt()
    data_root = args.data

    # 根据任务调整默认 input_size
    input_size = args.input_size
    if args.task == "classify" and args.input_size == 640:
        input_size = 224

    anno_json = args.anno_json or resolve_anno_json(data_root, args.task)

    with Profile("eval_bench") as perf:
        if args.task == "classify":
            from modelflow.pipelines import create_classify_pipeline
            from modelflow.datasets import ClassifyDataset
            from modelflow.evaluators import ClassifyEvaluator
            from modelflow.cfgs.imagenet import get_imagenet_classes

            class_list = get_imagenet_classes()
            pipeline = create_classify_pipeline(
                model_path=args.model,
                class_list=class_list,
                backend=args.backend,
                input_size=input_size,
            )
            dataset = ClassifyDataset(os.path.join(data_root, "val"), class_list)
            evaluator = ClassifyEvaluator(pipeline, dataset)

        elif args.task == "detect":
            from modelflow.pipelines import create_detect_pipeline
            from modelflow.datasets import COCODetectionDataset
            from modelflow.evaluators import DetectEvaluator
            from modelflow.cfgs.coco import class_list

            if args.backend == "triton":
                model_path = args.model  # Triton model name
            else:
                model_path = args.model

            pipeline = create_detect_pipeline(
                model_path=model_path,
                class_list=class_list,
                backend=args.backend,
                model_version=args.model_version,
                input_size=input_size,
                conf_thres=args.conf,
                iou_thres=args.iou,
            )
            dataset = COCODetectionDataset(
                os.path.join(data_root, "val2017"),
                class_list,
                anno_json=anno_json or None,
            )
            evaluator = DetectEvaluator(
                pipeline, dataset,
                gt_json=anno_json or None,
            )

        elif args.task == "segment":
            from modelflow.pipelines import create_segment_pipeline
            from modelflow.datasets import COCOSegmentDataset
            from modelflow.evaluators import SegmentEvaluator
            from modelflow.cfgs.coco import class_list

            pipeline = create_segment_pipeline(
                model_path=args.model,
                class_list=class_list,
                backend=args.backend,
                input_size=input_size,
                conf_thres=args.conf,
                iou_thres=args.iou,
            )
            dataset = COCOSegmentDataset(
                os.path.join(data_root, "val2017"),
                class_list,
                anno_json=anno_json or None,
            )
            evaluator = SegmentEvaluator(
                pipeline, dataset,
                gt_json=anno_json or None,
            )

        logger.info(f"Task: {args.task}, Backend: {args.backend}, Model: {args.model}")
        logger.info(f"Dataset: {data_root}, Input size: {input_size}")

        results = evaluator.run(save_pred_json=args.save_pred)
        logger.info(f"Results: {results}")

    print(f"⏱ Total time: {perf.elapsed:.2f}s")
    print(f"📊 Results: {results}")


if __name__ == "__main__":
    main()
