# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : eval_segment.py
@Author  : zj
@Description: 实例分割模型评估基准 — COCO segm mAP (YOLOv8-seg 等)

用法:
    # ONNX Runtime
    python3 samples/eval_segment.py \
        --model models/runtime/yolov8s-seg.onnx \
        --data /path/to/coco

    # TensorRT
    python3 samples/eval_segment.py \
        --model models/tensorrt/yolov8s-seg_fp16.engine \
        --backend tensorrt --data /path/to/coco

    # 另存预测 JSON
    python3 samples/eval_segment.py \
        --model models/runtime/yolov8s-seg.onnx \
        --data /path/to/coco --save-pred results.json

模型元数据和延迟测量请使用: python3 samples/parse_model.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelflow.pipelines import create_segment_pipeline
from eval import SegmentEvaluator
from data import build_dataset, get_class_names
from utils import Profile


def parse_opt():
    parser = argparse.ArgumentParser(description="ModelFlow 分割评估")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 Triton 模型名")
    parser.add_argument("--data", type=str, required=True,
                        help="COCO 数据集根目录（包含 annotations/ 和 val2017/）")
    parser.add_argument("--backend", type=str, default="onnxruntime",
                        choices=["onnxruntime", "triton"],
                        help="推理后端")
    parser.add_argument("--input-size", type=int, default=640,
                        help="输入尺寸")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="置信度阈值（评估用低阈值）")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU 阈值（segment 推荐 0.5）")
    parser.add_argument("--anno-json", type=str, default=None,
                        help="标注 JSON 路径（默认自动查找）")
    parser.add_argument("--save-pred", type=str, default=None,
                        help="预测结果 JSON 保存路径")
    return parser.parse_args()


def resolve_anno(data_root, anno_json):
    if anno_json:
        return anno_json
    for name in ("instances_val2017.json", "person_keypoints_val2017.json"):
        path = os.path.join(data_root, "annotations", name)
        if os.path.exists(path):
            return path
    return ""


def main():
    args = parse_opt()

    class_list = get_class_names("coco")
    anno_json = resolve_anno(args.data, args.anno_json)

    with Profile("pipeline.build"):
        pipeline = create_segment_pipeline(
            model_path=args.model,
            class_list=class_list,
            backend=args.backend,
            input_size=args.input_size,
            conf_thres=args.conf,
            iou_thres=args.iou,
        )

    dataset = build_dataset(
        "coco-seg", path=args.data, val="val2017",
        anno=f"annotations/{Path(anno_json).name}" if anno_json else None,
    )

    evaluator = SegmentEvaluator(pipeline, dataset, gt_json=anno_json or None)

    print(f"🔧 Task: segment, Backend: {args.backend}")
    print(f"📂 Data: {args.data}, Input size: {args.input_size}")
    print(f"📏 Conf: {args.conf}, IoU: {args.iou}")

    with Profile("eval") as perf:
        results = evaluator.run(save_pred_json=args.save_pred)

    print(f"⏱ Total time: {perf.elapsed:.2f}s")
    print(f"📊 Results: {results}")


if __name__ == "__main__":
    main()
