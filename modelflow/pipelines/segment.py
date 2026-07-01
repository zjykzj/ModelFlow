# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : segment.py
@Author  : zj
@Description: 实例分割 Pipeline 工厂

用法:
    pipeline = create_segment_pipeline(
        backend="onnxruntime",
        model_path="yolov8s-seg.onnx",
        class_list=["person", "car"],
    )
    result = pipeline(image)
    # result = {"boxes": ..., "scores": ..., "class_ids": ..., "masks": ...}
"""

from typing import List, Optional

from modelflow.interfaces import InferencePipeline, BaseBackend
from modelflow.processors.segment import SegmentPreprocessor, SegmentPostprocessor


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


def create_segment_pipeline(
    model_path: str,
    class_list: List[str],
    backend: str = "onnxruntime",
    processor: str = "numpy",
    input_size: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    half: bool = False,
    device: Optional[str] = None,
) -> InferencePipeline:
    """创建实例分割 Pipeline

    Args:
        model_path: 模型文件路径
        class_list: 类别名称列表
        backend: 后端类型
        processor: 处理器类型
        input_size: 输入尺寸
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值
        max_det: 最大检测数
        half: 是否使用半精度
        device: 设备

    Returns:
        InferencePipeline 实例
    """
    preprocessor = SegmentPreprocessor(input_size=input_size)

    _ensure_backend(backend)
    bk = _build_backend(backend, model_path, class_list,
                        task_type="segment", half=half, device=device)

    postprocessor = SegmentPostprocessor(
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        class_list=class_list,
        input_shape=(input_size, input_size),
    )

    return InferencePipeline(preprocessor, bk, postprocessor)
