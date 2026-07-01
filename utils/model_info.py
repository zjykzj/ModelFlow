# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : model_info.py
@Author  : zj
@Description: 模型元数据收集 — 文件大小 / 参数量 / FLOPs / 输入输出 shape

用法:
    from utils.model_info import get_model_info

    info = get_model_info("model.onnx")
    # -> {file_size_mb, param_count, flops, input_shapes, output_shapes, backend}

    info = get_model_info("model.engine")
    # -> TRT 引擎尝试读 .meta.json sidecar，无则只返回 file_size_mb
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

# protobuf 兼容性: 旧版 onnx + 新版 protobuf 需要此环境变量
# 必须在 import onnx 之前设置，所以放在模块级别
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _dim_value(dim) -> int:
    """兼容 ONNX Dimension 对象和纯整数两种格式。"""
    if hasattr(dim, "dim_value"):
        return dim.dim_value
    return int(dim)


def _shape_dims(shape_proto) -> List[int]:
    """从 TensorShapeProto 提取 dim 列表，-1 表示动态维度。"""
    return [_dim_value(d) if _dim_value(d) else -1 for d in shape_proto.dim]


def _get_onnx_model_info(model_path: str) -> dict:
    """从 ONNX 文件收集元数据。

    Args:
        model_path: .onnx 文件路径

    Returns:
        dict with: file_size_mb, param_count, flops, input_shapes, output_shapes, backend
    """
    import onnx

    model = onnx.load(model_path)
    graph = model.graph

    # --- 文件大小 ---
    file_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    # --- 参数量: 从 graph.initializer 累加 ---
    param_count = 0
    for tensor in graph.initializer:
        dims = [_dim_value(d) for d in tensor.dims]
        param_count += int(np.prod(dims))

    # --- FLOPs: 可选，基于 shape_inference + 节点类型计数 ---
    flops = _estimate_onnx_flops(model)

    # --- 输入输出 shape ---
    input_shapes = []
    for inp in graph.input:
        input_shapes.append(_shape_dims(inp.type.tensor_type.shape))

    output_shapes = []
    for out in graph.output:
        output_shapes.append(_shape_dims(out.type.tensor_type.shape))

    return {
        "model_path": model_path,
        "file_size_mb": round(file_size_mb, 2),
        "param_count": param_count,
        "flops": flops,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "backend": "onnxruntime",
    }


def _estimate_onnx_flops(model) -> Optional[int]:
    """估算 ONNX 模型的 FLOPs（基于 shape_inference + 节点类型）。

    对 Conv / Gemm / MatMul 等主力算子做 FLOPs 估算。
    如果 shape_inference 失败或依赖不足，返回 None。
    """
    try:
        from onnx.shape_inference import infer_shapes
        inferred = infer_shapes(model)
    except (ImportError, Exception):
        return None

    # 构建 name -> shape 映射（value_info + graph input/output）
    shape_map = {}
    for vi in inferred.graph.value_info:
        dims = _shape_dims(vi.type.tensor_type.shape)
        shape_map[vi.name] = dims

    for inp in inferred.graph.input:
        shape_map[inp.name] = _shape_dims(inp.type.tensor_type.shape)

    for out in inferred.graph.output:
        shape_map[out.name] = _shape_dims(out.type.tensor_type.shape)

    # 算子 FLOPs 系数（简化估算）
    OP_FLOPS_MAP = {
        "Conv": 2,      # MAC × 2 operations per MAC
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
    unknown_ops = set()

    for node in inferred.graph.node:
        op_type = node.op_type
        mul = OP_FLOPS_MAP.get(op_type)

        if mul is None:
            unknown_ops.add(op_type)
            continue

        # 尝试从输入 shape 估算计算量
        try:
            input_name = node.input[0]
            if input_name in shape_map:
                dims = shape_map[input_name]
                count = int(np.prod([d for d in dims if d and d > 0]))
                total_flops += count * mul
        except (IndexError, KeyError):
            pass

    # 如果未知算子太多，认为估算不可靠
    if unknown_ops:
        import logging
        logging.getLogger("utils.model_info").debug(
            f"Unknown op types for FLOPs: {sorted(unknown_ops)}"
        )

    return total_flops if total_flops > 0 else None


# ==================== TensorRT ====================


def _get_tensorrt_model_info(model_path: str) -> dict:
    """从 TensorRT 引擎文件收集元数据。

    引擎本身不透明，优先读 <engine>.meta.json sidecar，
    否则只返回文件大小。

    Args:
        model_path: .engine 文件路径

    Returns:
        dict with: file_size_mb, param_count, flops, input_shapes, output_shapes, backend
    """
    file_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    info = {
        "model_path": model_path,
        "file_size_mb": round(file_size_mb, 2),
        "param_count": None,
        "flops": None,
        "input_shapes": [],
        "output_shapes": [],
        "backend": "tensorrt",
    }

    # 1. 尝试读 sidecar
    meta_path = model_path + ".meta.json"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            info["param_count"] = meta.get("param_count")
            if "input_shape" in meta:
                info["input_shapes"] = [meta["input_shape"]]
            info["_source_onnx"] = meta.get("source_onnx")
        except (json.JSONDecodeError, IOError):
            pass

    # 2. 尝试从 TRT runtime 获取 I/O shape
    if not info["input_shapes"]:
        try:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(model_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = list(engine.get_tensor_shape(name))
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    info["input_shapes"].append([1 if d == -1 else d for d in shape])
                else:
                    info["output_shapes"].append([1 if d == -1 else d for d in shape])
        except ImportError:
            pass

    return info


# ==================== Unified Entry ====================


def get_model_info(model_path: str, backend: str = None) -> dict:
    """收集模型元数据（统一入口）。

    Args:
        model_path: 模型文件路径
        backend:    推理后端，None 则根据扩展名推断
                    .onnx -> onnxruntime, .engine -> tensorrt

    Returns:
        dict: {
            model_path, file_size_mb, param_count, flops,
            input_shapes, output_shapes, backend,
        }
        - param_count: int or None
        - flops: int or None (仅 ONNX 尝试估算)
        - input_shapes / output_shapes: List[List[int]]
    """
    if backend is None:
        if model_path.endswith(".onnx"):
            backend = "onnxruntime"
        elif model_path.endswith(".engine"):
            backend = "tensorrt"
        elif model_path.endswith((".plan", ".trt")):
            backend = "tensorrt"
        else:
            # 默认尝试 ONNX
            backend = "onnxruntime"

    if backend == "onnxruntime":
        return _get_onnx_model_info(model_path)
    elif backend == "tensorrt":
        return _get_tensorrt_model_info(model_path)
    elif backend == "triton":
        # Triton 是远程模型，本地无法获取元数据
        return {
            "model_path": model_path,
            "file_size_mb": None,
            "param_count": None,
            "flops": None,
            "input_shapes": [],
            "output_shapes": [],
            "backend": "triton",
        }
    else:
        raise ValueError(
            f"Unsupported backend: {backend} for model info. "
            f"Supported: onnxruntime, tensorrt, triton."
        )
