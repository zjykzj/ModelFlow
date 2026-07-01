# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : classify.py
@Author  : zj
@Description: 分类 Pipeline 工厂

用法:
    pipeline = create_classify_pipeline(
        backend="onnxruntime",
        model_path="efficientnet_b0.onnx",
        class_list=["cat", "dog", ...],
    )
    result = pipeline(image)
"""

from typing import List, Optional

from modelflow.interfaces import InferencePipeline, BaseBackend
from modelflow.processors.classify import ClassifyPreprocessor, ClassifyPostprocessor


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


def create_classify_pipeline(
    model_path: str,
    class_list: List[str],
    backend: str = "onnxruntime",
    processor: str = "numpy",
    input_size: int = 224,
    topk: int = 5,
    half: bool = False,
    device: Optional[str] = None,
) -> InferencePipeline:
    """创建分类 Pipeline

    Args:
        model_path: 模型文件路径
        class_list: 类别名称列表
        backend: 后端类型（"onnxruntime" / "tensorrt" / "triton"）
        processor: 处理器类型（"numpy"）
        input_size: 输入尺寸（默认 224）
        topk: 返回 top-k 结果
        half: 是否使用半精度
        device: 设备

    Returns:
        InferencePipeline 实例
    """
    preprocessor = ClassifyPreprocessor(crop_size=input_size)

    _ensure_backend(backend)
    bk = _build_backend(backend, model_path, class_list,
                        task_type="classify", half=half, device=device)

    postprocessor = ClassifyPostprocessor(topk=topk, class_list=class_list)

    return InferencePipeline(preprocessor, bk, postprocessor)
