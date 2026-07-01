# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : semantic_seg.py
@Author  : zj
@Description: 语义分割 Pipeline 工厂
"""

from typing import List, Optional, Tuple

from modelflow.interfaces import InferencePipeline, BaseBackend
from modelflow.processors.semantic_seg import (
    SemanticSegPreprocessor,
    SemanticSegPostprocessor,
)


def _ensure_backend(name: str) -> None:
    """Lazy-import the requested backend module (side-effect only)."""
    if name == "onnxruntime":
        from modelflow.backends.onnx import OnnxBackend  # noqa: F401
    elif name == "tensorrt":
        from modelflow.backends.tensorrt import TensorrtBackend  # noqa: F401
    elif name == "triton":
        from modelflow.backends.triton import TritonBackend  # noqa: F401


def _build_backend(name: str, model_path: str, class_list: List[str],
                   task_type: str, half: bool, device: Optional[str]) -> BaseBackend:
    """Direct construction — no Registry indirection."""
    if name == "onnxruntime":
        from modelflow.backends.onnx import OnnxBackend
        return OnnxBackend(model_path, class_list, task_type=task_type,
                           half=half, device=device)
    elif name == "tensorrt":
        from modelflow.backends.tensorrt import TensorrtBackend
        return TensorrtBackend(model_path, class_list, task_type=task_type,
                               half=half, device=device)
    elif name == "triton":
        from modelflow.backends.triton import TritonBackend
        return TritonBackend(model_path, class_list, task_type=task_type,
                             half=half, device=device)
    raise ValueError(f"Unsupported backend: {name}")


def create_semantic_seg_pipeline(
    model_path: str,
    class_list: List[str],
    backend: str = "onnxruntime",
    input_size: Optional[Tuple[int, int]] = None,
    half: bool = False,
    device: Optional[str] = None,
) -> InferencePipeline:
    """创建语义分割 Pipeline

    Args:
        model_path: 模型路径
        class_list: 类别名称列表
        backend: 后端类型
        input_size: 输入尺寸 (H, W)，None 表示保持原图
        half: 是否使用半精度
        device: 设备

    Returns:
        InferencePipeline 实例
    """
    preprocessor = SemanticSegPreprocessor(target_size=input_size)

    _ensure_backend(backend)
    bk = _build_backend(backend, model_path, class_list,
                        task_type="semantic_seg", half=half, device=device)

    postprocessor = SemanticSegPostprocessor()

    return InferencePipeline(preprocessor, bk, postprocessor)
