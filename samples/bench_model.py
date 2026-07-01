# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/17
@File    : bench_model.py
@Author  : zj
@Description: 模型延迟基准测试 — 后端推理 / 完整管线

支持 ONNX Runtime / TensorRT / Triton 三种后端。
与 analyze_model.py 配合：先用 analyze 拿到架构常数，再用 bench 对比不同后端的延迟。

用法:
    # 后端推理延迟（dummy tensor，跳过 pre/post）—— 用于对比不同后端的纯推理速度
    python3 samples/bench_model.py --model models/yolov8s.onnx --task detect

    # 完整管线延迟（随机图，pre + infer + post 三段计时）
    python3 samples/bench_model.py --model models/yolov8s.onnx --task detect --mode pipeline

    # 完整管线 + 真实图
    python3 samples/bench_model.py --model models/yolov8s.onnx --task detect --mode pipeline --image assets/bus.jpg

    # TensorRT 后端
    python3 samples/bench_model.py --model models/tensorrt/yolov8s_fp16.engine --backend tensorrt --task detect

    # Triton 后端
    python3 samples/bench_model.py --model Detect_COCO_YOLOv8s_ONNX --backend triton --task detect

    # 自定义迭代
    python3 samples/bench_model.py --model models/yolov8s.onnx --task detect --iters 500 --warmup 50

    # 导出 JSON
    python3 samples/bench_model.py --model models/yolov8s.onnx --task detect --json latency.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.profile import measure_inference_latency, measure_pipeline_latency


def parse_opt():
    parser = argparse.ArgumentParser(description="ModelFlow 模型延迟基准测试")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径（.onnx / .engine）或 Triton 模型名")
    parser.add_argument("--backend", type=str, default="onnxruntime",
                        choices=["onnxruntime", "tensorrt", "triton"],
                        help="推理后端（默认 onnxruntime）")
    parser.add_argument("--task", type=str, default="detect",
                        choices=["classify", "detect", "segment", "semantic_seg"],
                        help="任务类型（默认 detect）")
    parser.add_argument("--model-version", type=str, default="v8",
                        choices=["v5", "v8", "v11"],
                        help="YOLO 版本（仅 detect/segment 使用）")
    parser.add_argument("--input-size", type=int, default=None,
                        help="输入尺寸（默认从 ONNX 推断或 640）")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值（pipeline 模式）")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值（pipeline 模式）")

    # 测量模式
    parser.add_argument("--mode", type=str, default="backend",
                        choices=["backend", "pipeline"],
                        help="backend: 纯推理延迟（dummy tensor）；"
                             "pipeline: 完整管线（pre+infer+post）")
    parser.add_argument("--image", type=str, default=None,
                        help="pipeline 模式下的输入图片（未指定则用随机图）")
    parser.add_argument("--iters", type=int, default=100,
                        help="测量迭代次数（默认 100）")
    parser.add_argument("--warmup", type=int, default=10,
                        help="预热迭代次数（默认 10）")

    # 输出
    parser.add_argument("--json", type=str, default=None,
                        help="将结果保存为 JSON 文件")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _resolve_input_size(model_path: str, cli_value: Optional[int]) -> int:
    """解析输入尺寸：CLI 参数 > ONNX 自动推断 > 默认 640。"""
    if cli_value is not None:
        return cli_value
    if model_path.endswith(".onnx") and os.path.isfile(model_path):
        sz = _read_input_size_from_onnx(model_path)
        if sz is not None:
            return sz
    return 640


def _read_input_size_from_onnx(model_path: str) -> Optional[int]:
    """从 ONNX 文件快速读取输入尺寸（轻量级，不解析全图）。"""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        import onnx
        model = onnx.load(model_path)
        for inp in model.graph.input:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                v = d.dim_value if hasattr(d, "dim_value") else int(d)
                dims.append(v if v else -1)
            if len(dims) >= 4:
                return dims[2]
    except Exception:
        pass
    return None


def _build_pipeline(model_path: str, backend: str, task: str,
                    input_size: int, model_version: str,
                    conf: float, iou: float):
    """根据任务类型构建推理管线。"""
    dummy_classes = ["class_0"]

    if task == "classify":
        from modelflow.pipelines import create_classify_pipeline
        return create_classify_pipeline(
            model_path=model_path, class_list=dummy_classes,
            backend=backend, input_size=input_size,
        )
    elif task == "detect":
        from modelflow.pipelines import create_detect_pipeline
        return create_detect_pipeline(
            model_path=model_path, class_list=dummy_classes,
            backend=backend, input_size=input_size,
            conf_thres=conf, iou_thres=iou,
            model_version=model_version,
        )
    elif task == "segment":
        from modelflow.pipelines import create_segment_pipeline
        return create_segment_pipeline(
            model_path=model_path, class_list=dummy_classes,
            backend=backend, input_size=input_size,
            conf_thres=conf, iou_thres=iou,
        )
    elif task == "semantic_seg":
        from modelflow.pipelines import create_semantic_seg_pipeline
        return create_semantic_seg_pipeline(
            model_path=model_path, class_list=dummy_classes,
            backend=backend, input_size=(input_size, input_size),
        )
    else:
        raise ValueError(f"Unsupported task: {task}")


# ═══════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════

def _sep(char="=", width=62):
    print(char * width)


def _section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def _kv(key: str, value: str, indent: int = 2):
    print(f"{' ' * indent}{key:<22} {value}")


def _print_latency(result: dict):
    """格式化打印延迟测量结果。"""
    mode = result.get("mode", "N/A")
    name = Path(result.get("model_path", "")).name or result.get("model_path", "")

    _sep()
    print(f"  {name}  —  Latency Benchmark")
    _sep()

    _section("Setup")
    _kv("Model", result.get("model_path", "N/A"))
    _kv("Backend", result.get("backend", "N/A"))
    _kv("Task", result.get("task", "N/A"))
    if result.get("model_version"):
        _kv("Model version", result["model_version"])
    _kv("Mode", mode.replace("_", " "))
    _kv("Input shape", str(result.get("input_shape", "N/A")))

    _section("Latency")
    if mode == "full_pipeline":
        for stage in ("preprocess", "inference", "postprocess", "total"):
            s = result.get(stage, {})
            if s:
                _kv(f"{stage.capitalize()} (ms)",
                    f"mean={s['mean_ms']:.2f}  p95={s['p95_ms']:.2f}")
    else:
        _kv("Inference (ms)",
            f"mean={result.get('mean_ms', 0):.2f}  "
            f"p95={result.get('p95_ms', 0):.2f}")

    _kv("Throughput", f"{result.get('fps', 'N/A')} FPS")
    _kv("Iterations",
        f"{result.get('iters', 'N/A')} "
        f"(warmup={result.get('warmup', 'N/A')})")
    print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_opt()

    if not args.model:
        print("❌ --model is required")
        sys.exit(1)

    # 本地文件存在性检查（Triton 模型名跳过）
    if args.backend != "triton" and not os.path.isfile(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    input_size = _resolve_input_size(args.model, args.input_size)

    # ── Build pipeline ──
    print(f"\n⏳ Building pipeline ({args.task}, {args.backend})...")
    try:
        pipeline = _build_pipeline(
            model_path=args.model, backend=args.backend,
            task=args.task, input_size=input_size,
            model_version=args.model_version,
            conf=args.conf, iou=args.iou,
        )
    except Exception as e:
        print(f"❌ Failed to build pipeline: {e}")
        sys.exit(1)

    # ── Measure ──
    if args.mode == "pipeline":
        if args.image:
            import cv2
            image = cv2.imread(args.image)
            if image is None:
                print(f"❌ Cannot read image: {args.image}")
                sys.exit(1)
        else:
            # 随机图（模拟真实尺寸）
            image = np.random.randint(
                0, 256, (input_size, input_size, 3), dtype=np.uint8
            )

        print(f"⏱  Measuring full pipeline latency"
              f" ({args.iters} iters, warmup={args.warmup})...")

        result = measure_pipeline_latency(
            pipeline, image,
            warmup=args.warmup, iters=args.iters,
            conf_thres=args.conf, iou_thres=args.iou,
        )
    else:
        input_shape = (1, 3, input_size, input_size)
        print(f"⏱  Measuring backend inference latency"
              f" ({args.iters} iters, warmup={args.warmup})...")

        result = measure_inference_latency(
            pipeline, input_shape=input_shape,
            warmup=args.warmup, iters=args.iters,
        )

    # ── Attach metadata ──
    result["model_path"] = args.model
    result["backend"] = args.backend
    result["task"] = args.task
    if args.task in ("detect", "segment"):
        result["model_version"] = args.model_version

    # ── Output ──
    _print_latency(result)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"💾 Results saved to: {args.json}")


if __name__ == "__main__":
    main()
