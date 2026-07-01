# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/17
@File    : analyze_model.py
@Author  : zj
@Description: 模型架构分析 — 参数量 / FLOPs / I/O shape / 推理配置推断

仅支持 ONNX 模型。不需要推理运行时或 GPU。适用于模型选型和文档生成。

用法:
    # 检测模型
    python3 samples/analyze_model.py --model models/yolov8s.onnx

    # 分割模型
    python3 samples/analyze_model.py --model models/yolov8s-seg.onnx

    # 分类模型
    python3 samples/analyze_model.py --model models/efficientnet_b0.onnx

    # 保存为 JSON（方便脚本消费）
    python3 samples/analyze_model.py --model models/yolov8s.onnx --json meta.json

输出会推断 task_type / model_version / num_classes，并给出建议的 CLI 命令。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def parse_opt():
    parser = argparse.ArgumentParser(description="ModelFlow 模型架构分析（仅 ONNX）")
    parser.add_argument("--model", type=str, required=True,
                        help="ONNX 模型文件路径（必须 .onnx）")
    parser.add_argument("--json", type=str, default=None,
                        help="将分析结果保存为 JSON 文件")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# ONNX metadata parsing
# ═══════════════════════════════════════════════════════════════

def _dim_value(dim) -> int:
    """兼容 ONNX Dimension 对象和纯整数两种格式。"""
    if hasattr(dim, "dim_value"):
        return dim.dim_value
    return int(dim)


def _shape_dims(shape_proto) -> List[int]:
    """从 TensorShapeProto 提取 dim 列表，-1 表示动态维度。"""
    return [_dim_value(d) if _dim_value(d) else -1 for d in shape_proto.dim]


def _get_onnx_metadata(model_path: str) -> dict:
    """从 ONNX 文件收集架构元数据。

    Returns:
        dict with: model_path, file_size_mb, param_count, flops,
                   input_shapes, output_shapes, opset_version,
                   producer_name, ir_version
    """
    import onnx

    model = onnx.load(model_path)
    graph = model.graph

    # 文件大小
    file_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    # 参数量：从 graph.initializer 累加
    param_count = 0
    for tensor in graph.initializer:
        dims = [_dim_value(d) for d in tensor.dims]
        param_count += int(np.prod(dims))

    # FLOPs 估算
    flops = _estimate_flops(model)

    # 输入输出 shape
    input_shapes = []
    for inp in graph.input:
        input_shapes.append(_shape_dims(inp.type.tensor_type.shape))

    output_shapes = []
    for out in graph.output:
        output_shapes.append(_shape_dims(out.type.tensor_type.shape))

    # 附加信息
    opset_version = model.opset_import[0].version if model.opset_import else None
    producer = model.producer_name or None

    return {
        "model_path": model_path,
        "file_size_mb": round(file_size_mb, 2),
        "param_count": param_count,
        "flops": flops,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "opset_version": opset_version,
        "producer_name": producer,
    }


def _estimate_flops(model) -> Optional[int]:
    """基于 shape_inference + 节点类型估算 FLOPs。

    对 Conv / Gemm / MatMul 等主力算子做 MAC → FLOPs 换算。
    shape_inference 失败或依赖不足时返回 None。
    """
    try:
        from onnx.shape_inference import infer_shapes
        inferred = infer_shapes(model)
    except (ImportError, Exception):
        return None

    # 构建 name → shape 映射
    shape_map = {}
    for vi in inferred.graph.value_info:
        shape_map[vi.name] = _shape_dims(vi.type.tensor_type.shape)
    for inp in inferred.graph.input:
        shape_map[inp.name] = _shape_dims(inp.type.tensor_type.shape)
    for out in inferred.graph.output:
        shape_map[out.name] = _shape_dims(out.type.tensor_type.shape)

    OP_FLOPS_MAP = {
        "Conv": 2,
        "Gemm": 2,
        "MatMul": 2,
        "BatchNormalization": 1,
        "Relu": 1,
        "Sigmoid": 4,
        "Tanh": 6,
        "Softmax": 5,
        "MaxPool": 1,
        "AveragePool": 1,
        "GlobalAveragePool": 1,
    }

    total_flops = 0
    for node in inferred.graph.node:
        mul = OP_FLOPS_MAP.get(node.op_type)
        if mul is None:
            continue
        try:
            input_name = node.input[0]
            if input_name in shape_map:
                dims = shape_map[input_name]
                count = int(np.prod([d for d in dims if d and d > 0]))
                total_flops += count * mul
        except (IndexError, KeyError):
            pass

    return total_flops if total_flops > 0 else None


# ═══════════════════════════════════════════════════════════════
# Config inference from I/O shapes
# ═══════════════════════════════════════════════════════════════

# segment 任务的 proto mask 输出特征: (1, 32, 160, 160)
_PROTO_MASK_PATTERN = (32, 160, 160)


def infer_config(output_shapes: List[List[int]],
                 input_shapes: List[List[int]]) -> dict:
    """从 I/O shape 推断推理配置。

    推断规则（按优先级）:
    1. 有 (1,32,160,160) proto mask → segment (v8)
    2. 2D 输出 (1, N)             → classify
    3. 4D 输出 (1, C, H, W)       → semantic_seg
    4. 3D 输出 (1, C, G):
       - C < G                    → detect v8/v11, nc = C - 4
       - C > G                    → detect v5,      nc = G - 5

    Returns:
        dict with: task_type, model_version, num_classes,
                   input_size, class_source
    """
    if not output_shapes:
        return _unknown_config(input_shapes)

    main_out = output_shapes[0]
    ndim = len(main_out)

    input_size = None
    if input_shapes and len(input_shapes[0]) >= 4:
        input_size = input_shapes[0][2]

    # ── 1. Proto mask → segment ──
    for out in output_shapes[1:]:
        if len(out) == 4:
            if (out[1], out[2], out[3]) == _PROTO_MASK_PATTERN:
                nc = main_out[1] - 4 - 32  # bbox(4) + mask_coeffs(32)
                return {
                    "task_type": "segment",
                    "model_version": "v8",
                    "num_classes": nc,
                    "input_size": input_size,
                    "class_source": "coco" if nc == 80 else None,
                }

    # ── 2. 2D output (1, N) → classify ──
    if ndim == 2:
        nc = main_out[1] if len(main_out) > 1 else main_out[0]
        return {
            "task_type": "classify",
            "model_version": None,
            "num_classes": nc,
            "input_size": input_size,
            "class_source": "imagenet" if nc == 1000 else None,
        }

    # ── 3. 4D spatial output → semantic_seg ──
    if ndim == 4 and main_out[2] > 1 and main_out[3] > 1:
        return {
            "task_type": "semantic_seg",
            "model_version": None,
            "num_classes": main_out[1],
            "input_size": input_size,
            "class_source": None,
        }

    # ── 4. 3D output → detect ──
    if ndim == 3:
        d1, d2 = main_out[1], main_out[2]

        if d1 < d2:
            # v8/v11: (1, 84, 8400) → channels < grid
            model_version = "v8"
            nc = d1 - 4
        else:
            # v5: (1, 25200, 85) → anchors > channels
            model_version = "v5"
            nc = d2 - 5

        return {
            "task_type": "detect",
            "model_version": model_version,
            "num_classes": nc,
            "input_size": input_size,
            "class_source": "coco" if nc == 80 else None,
        }

    return _unknown_config(input_shapes)


def _unknown_config(input_shapes):
    input_size = None
    if input_shapes and len(input_shapes[0]) >= 4:
        input_size = input_shapes[0][2]
    return {
        "task_type": "unknown",
        "model_version": None,
        "num_classes": None,
        "input_size": input_size,
        "class_source": None,
    }


# ═══════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════

def _fmt_size(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    if n >= 1e9:
        return f"{n / 1e9:.2f} G"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} K"
    return str(n)


def _fmt_flops(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"{n / 1e12:.2f} T"
    if n >= 1e9:
        return f"{n / 1e9:.2f} G"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    return str(n)


def _sep(char="=", width=62):
    print(char * width)


def _section(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def _kv(key: str, value: str, indent: int = 2):
    print(f"{' ' * indent}{key:<22} {value}")


def _print_analysis(meta: dict, config: dict):
    """格式化打印架构分析结果。"""
    name = Path(meta["model_path"]).name

    # ── Header ──
    _sep()
    print(f"  {name}  —  Architecture Analysis")
    _sep()

    # ── File Info ──
    _section("File Info")
    _kv("Model path", meta["model_path"])
    _kv("File size", f"{meta['file_size_mb']} MB")
    _kv("Producer", meta.get("producer_name") or "N/A")
    _kv("Opset version", str(meta.get("opset_version") or "N/A"))

    # ── Model Stats ──
    _section("Model Stats")
    _kv("Parameters", _fmt_size(meta.get("param_count")))
    flops_note = " (rough estimate)" if meta.get("flops") else ""
    _kv("FLOPs", _fmt_flops(meta.get("flops")) + flops_note)

    # ── I/O Shapes ──
    _section("I/O Shapes")
    for i, s in enumerate(meta.get("input_shapes", [])):
        _kv(f"Input #{i}", str(s))
    for i, s in enumerate(meta.get("output_shapes", [])):
        _kv(f"Output #{i}", str(s))

    # ── Inferred Config ──
    _section("Inferred Config")
    _kv("Task type", config.get("task_type", "N/A"))
    _kv("Model version", config.get("model_version") or "N/A")
    _kv("Input size", str(config.get("input_size") or "N/A"))
    _kv("Num classes", str(config.get("num_classes") or "N/A"))

    class_src = config.get("class_source")
    if class_src:
        _kv("Class source", f"{class_src} (auto-detected)")
    else:
        _kv("Class source", "unknown — provide --classes manually")

    # ── Suggested CLI ──
    task = config.get("task_type", "detect")
    version = config.get("model_version")
    input_size = config.get("input_size") or 640
    nc = config.get("num_classes")

    _section("Suggested CLI")
    flags = f"--task {task} --model {meta['model_path']} --input-size {input_size}"
    if version:
        flags += f" --model-version {version}"
    if task in ("detect", "segment") and nc == 80:
        flags += " --classes coco"
    elif task == "classify" and nc == 1000:
        flags += " --classes imagenet"
    else:
        flags += " --classes <your-class-file>"

    print(f"  python3 samples/infer.py   {flags} --image <image>")
    print(f"  python3 samples/bench_model.py --model {meta['model_path']}"
          f" --task {task}" + (f" --model-version {version}" if version else ""))
    print(f"  python3 samples/eval_{task}.py  {flags} --data <dataset-path>")
    print()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_opt()

    if not args.model.endswith(".onnx"):
        print("❌ analyze_model only supports .onnx files.")
        print("   For latency on .engine or Triton models, use bench_model.py.")
        print("   To inspect an .engine, run analyze_model on its source .onnx first.")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    # ── Parse ──
    try:
        meta = _get_onnx_metadata(args.model)
    except Exception as e:
        print(f"❌ Failed to parse ONNX: {e}")
        sys.exit(1)

    # ── Infer ──
    config = infer_config(meta["output_shapes"], meta["input_shapes"])

    # ── Output ──
    _print_analysis(meta, config)

    if args.json:
        result = {**meta, "inferred": config}
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"💾 Analysis saved to: {args.json}")


if __name__ == "__main__":
    main()
