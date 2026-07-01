# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : infer.py
@Author  : zj
@Description: 统一推理入口 — 分类/检测/分割，ONNX Runtime / TensorRT / Triton

用法:
    # 检测
    python3 samples/infer.py --task detect --model models/runtime/yolov8s.onnx --image assets/bus.jpg

    # 分类
    python3 samples/infer.py --task classify --model models/runtime/efficientnet_b0.onnx --image assets/bus.jpg --classes imagenet --input-size 224

    # 分割
    python3 samples/infer.py --task segment --model models/runtime/yolov8s-seg.onnx --image assets/bus.jpg

    # 语义分割
    python3 samples/infer.py --task semantic_seg --model models/runtime/segformer.onnx --image assets/bus.jpg

    # 保存 + 显示
    python3 samples/infer.py --task detect --model models/runtime/yolov8s.onnx --image assets/bus.jpg --save result.jpg --show
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelflow.config import COCO_CLASSES as coco_classes
from utils import Profile
from samples.utils import draw_classification, draw_detections, draw_segmentation


def parse_opt():
    parser = argparse.ArgumentParser(description="ModelFlow 统一推理入口")
    parser.add_argument("--task", type=str, default="detect",
                        choices=["classify", "detect", "segment", "semantic_seg"],
                        help="任务类型")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 Triton 模型名")
    parser.add_argument("--image", type=str, required=True,
                        help="输入图片路径")
    parser.add_argument("--backend", type=str, default="onnxruntime",
                        choices=["onnxruntime", "tensorrt", "triton"],
                        help="推理后端")
    parser.add_argument("--classes", type=str, default=None,
                        help="类别来源: 'coco' / 'imagenet' / 文件路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (detect/segment)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU 阈值 (detect/segment)")
    parser.add_argument("--input-size", type=int, default=640,
                        help="输入尺寸 (classify 默认 224)")
    parser.add_argument("--model-version", type=str, default="v8",
                        choices=["v5", "v8", "v11"],
                        help="YOLO 版本 (detect)")
    parser.add_argument("--save", type=str, default=None,
                        help="可视化结果保存路径")
    parser.add_argument("--show", action="store_true",
                        help="弹窗显示结果")
    return parser.parse_args()


def load_class_list(classes_arg):
    if classes_arg == "coco":
        from modelflow.config import COCO_CLASSES
        return COCO_CLASSES
    elif classes_arg == "imagenet":
        from modelflow.config import get_imagenet_classes
        return get_imagenet_classes()
    elif classes_arg and os.path.isfile(classes_arg):
        with open(classes_arg) as f:
            return [line.strip() for line in f if line.strip()]
    return []


def main():
    args = parse_opt()
    class_list = load_class_list(args.classes)

    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Cannot read image: {args.image}")
        sys.exit(1)

    print(f"📷 Image: {args.image} ({img.shape[1]}x{img.shape[0]})")
    print(f"🔧 Task: {args.task}, Backend: {args.backend}, Model: {args.model}")

    # 构建 Pipeline
    with Profile("pipeline.build"):
        input_size = args.input_size
        if args.task == "classify" and args.input_size == 640:
            input_size = 224

        if args.task == "classify":
            from modelflow.pipelines import create_classify_pipeline
            pipeline = create_classify_pipeline(
                model_path=args.model,
                class_list=class_list or ["unknown"],
                backend=args.backend,
                input_size=input_size,
            )
        elif args.task == "detect":
            from modelflow.pipelines import create_detect_pipeline
            pipeline = create_detect_pipeline(
                model_path=args.model,
                class_list=class_list or coco_classes,
                backend=args.backend,
                input_size=input_size,
                conf_thres=args.conf,
                iou_thres=args.iou,
                model_version=args.model_version,
            )
        elif args.task == "segment":
            from modelflow.pipelines import create_segment_pipeline
            pipeline = create_segment_pipeline(
                model_path=args.model,
                class_list=class_list or coco_classes,
                backend=args.backend,
                input_size=input_size,
                conf_thres=args.conf,
                iou_thres=args.iou,
            )
        else:
            from modelflow.pipelines import create_semantic_seg_pipeline
            pipeline = create_semantic_seg_pipeline(
                model_path=args.model,
                class_list=class_list or ["unknown"],
                backend=args.backend,
                input_size=(input_size, input_size),
            )

    # 推理
    with Profile("inference") as perf:
        result = pipeline(img, conf_thres=args.conf, iou_thres=args.iou)

    # 可视化
    if args.task == "classify":
        names = result.get("class_names", [])
        scores = result.get("scores", [])
        for name, score in zip(names, scores):
            print(f"  🏷 {name}: {score:.4f}")
        vis = draw_classification(img, names, scores)

    elif args.task == "semantic_seg":
        class_map = result.get("class_map")
        print(f"  🗺 Class map shape: {class_map.shape}")
        # 语义分割：argmax 着色
        colormap = result.get("colormap")
        if colormap is not None:
            vis = colormap
        else:
            vis = img

    else:  # detect / segment
        boxes = result.get("boxes", [])
        print(f"  📦 Detected {len(boxes)} objects")
        for i in range(min(10, len(boxes))):
            x1, y1, x2, y2 = map(int, boxes[i])
            label = result.get("class_names", [""])[i] if result.get("class_names") else str(result["class_ids"][i])
            print(f"    [{i}] {label}: ({x1},{y1})-({x2},{y2}) score={result['scores'][i]:.4f}")
        if len(boxes) > 10:
            print(f"    ... and {len(boxes) - 10} more")

        if args.task == "segment" and "masks" in result and result["masks"] is not None:
            vis = draw_segmentation(
                img, boxes, result["masks"], result["scores"],
                result["class_ids"], class_list,
            )
        else:
            vis = draw_detections(
                img, boxes, result["scores"],
                result["class_ids"], class_list,
            )

    print(f"  ⏱ Time: {perf.elapsed * 1000:.1f}ms")

    # 保存 / 显示
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        cv2.imwrite(args.save, vis)
        print(f"  💾 Saved to: {args.save}")

    if args.show:
        cv2.imshow("ModelFlow Inference", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
