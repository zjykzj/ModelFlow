# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : infer.py
@Author  : zj
@Description: 统一推理入口（使用 modelflow）

支持分类/检测/分割任务，ONNX Runtime / TensorRT / Triton 三种后端。

用法:
    # 检测 (YOLOv8 ONNX)
    python3 samples/infer.py --task detect --model yolov8s.onnx --image bus.jpg

    # 分类 (EfficientNet)
    python3 samples/infer.py --task classify --model efficientnet_b0.onnx --image image.jpg

    # 分割 (YOLOv8-seg TensorRT)
    python3 samples/infer.py --task segment --model yolov8s-seg.engine --backend tensorrt --image bus.jpg

    # Triton
    python3 samples/infer.py --task detect --model Detect_COCO_YOLOv8s_ONNX --backend triton --image bus.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelflow.cfgs.coco import class_list as coco_classes
from modelflow.utils import Profile
from modelflow.viz import DetectVisualizer


def parse_opt():
    parser = argparse.ArgumentParser(description="ModelFlow 统一推理入口")
    parser.add_argument("--task", type=str, default="detect",
                        choices=["classify", "detect", "segment", "semantic_seg"],
                        help="任务类型")
    parser.add_argument("--model", type=str, required=True, help="模型路径或 Triton 模型名")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--backend", type=str, default="onnxruntime",
                        choices=["onnxruntime", "tensorrt", "triton"],
                        help="推理后端")
    parser.add_argument("--classes", type=str, default=None,
                        help="类别列表文件或 'coco' / 'imagenet'")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU 阈值")
    parser.add_argument("--input-size", type=int, default=640, help="输入尺寸")
    parser.add_argument("--save", type=str, default=None, help="结果保存路径")
    parser.add_argument("--model-version", type=str, default="v8",
                        choices=["v5", "v8", "v11"], help="YOLO 模型版本")
    return parser.parse_args()


def load_class_list(classes_arg):
    if classes_arg == "coco":
        from modelflow.cfgs.coco import class_list
        return class_list
    elif classes_arg == "imagenet":
        from modelflow.cfgs.imagenet import get_imagenet_classes
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
        if args.task == "classify":
            from modelflow.pipelines import create_classify_pipeline
            pipeline = create_classify_pipeline(
                model_path=args.model, class_list=class_list or ["unknown"],
                backend=args.backend, input_size=args.input_size,
            )
        elif args.task == "detect":
            from modelflow.pipelines import create_detect_pipeline
            pipeline = create_detect_pipeline(
                model_path=args.model, class_list=class_list or coco_classes,
                backend=args.backend, input_size=args.input_size,
                conf_thres=args.conf, iou_thres=args.iou,
                model_version=args.model_version,
            )
        elif args.task == "segment":
            from modelflow.pipelines import create_segment_pipeline
            pipeline = create_segment_pipeline(
                model_path=args.model, class_list=class_list or coco_classes,
                backend=args.backend, input_size=args.input_size,
                conf_thres=args.conf, iou_thres=args.iou,
            )
        else:
            from modelflow.pipelines import create_semantic_seg_pipeline
            pipeline = create_semantic_seg_pipeline(
                model_path=args.model, class_list=class_list or ["unknown"],
                backend=args.backend, input_size=(args.input_size, args.input_size),
            )

    # 推理
    with Profile("inference") as perf:
        result = pipeline(img, conf_thres=args.conf, iou_thres=args.iou)

    # 输出结果摘要
    if args.task == "classify":
        for name, score in zip(result.get("class_names", []), result.get("scores", [])):
            print(f"  🏷 {name}: {score:.4f}")
    elif args.task in ("detect", "segment"):
        boxes = result.get("boxes", [])
        print(f"  📦 Detected {len(boxes)} objects")
        for i in range(min(5, len(boxes))):
            x1, y1, x2, y2 = map(int, boxes[i])
            label = result.get("class_names", [""])[i] if result.get("class_names") else result["class_ids"][i]
            print(f"    [{i}] {label}: ({x1},{y1})-({x2},{y2}) score={result['scores'][i]:.4f}")
    elif args.task == "semantic_seg":
        print(f"  🗺 Class map shape: {result['class_map'].shape}")
        if 'colormap' in result:
            print(f"  🎨 Colormap shape: {result['colormap'].shape}")

    # 保存结果
    save_path = args.save or f"output_{Path(args.image).stem}.jpg"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if args.task in ("detect", "segment") and len(result.get("boxes", [])) > 0:
        visualizer = DetectVisualizer(class_list)
        annotated = visualizer.draw(img, result)
        cv2.imwrite(save_path, annotated)
        print(f"  ✅ Result saved to {save_path}")
    elif args.task == "semantic_seg" and "colormap" in result:
        cv2.imwrite(save_path, result["colormap"])
        print(f"  ✅ Segmentation mask saved to {save_path}")
    else:
        print(f"  ⚠️ No result to visualize")


if __name__ == "__main__":
    main()
