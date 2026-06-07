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

from modelflow.core.interfaces import InferencePipeline
from modelflow.core.config import ModelConfig
from modelflow.backends.onnx import OnnxBackend
from modelflow.processors.segment import SegmentPreprocessor, SegmentPostprocessor


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

    if backend == "onnxruntime":
        bk = OnnxBackend(model_path, class_list, task_type="segment",
                         half=half, device=device)
    elif backend == "tensorrt":
        from modelflow.backends.tensorrt import TensorrtBackend
        bk = TensorrtBackend(model_path, class_list, task_type="segment",
                             half=half, device=device)
    elif backend == "triton":
        from modelflow.backends.triton import TritonBackend
        bk = TritonBackend(model_path, class_list, task_type="segment",
                           half=half, device=device)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    postprocessor = SegmentPostprocessor(
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        class_list=class_list,
        input_shape=(input_size, input_size),
    )

    return InferencePipeline(preprocessor, bk, postprocessor)


def create_segment_pipeline_from_config(config: ModelConfig) -> InferencePipeline:
    """从 ModelConfig 创建实例分割 Pipeline"""
    return create_segment_pipeline(
        model_path=config.model_path,
        class_list=config.class_list,
        backend=config.backend,
        processor=config.processor,
        input_size=config.input_size,
        conf_thres=config.conf_thres,
        iou_thres=config.iou_thres,
        max_det=config.max_det,
        half=config.half,
        device=config.device,
    )
