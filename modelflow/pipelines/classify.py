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

from modelflow.core.interfaces import InferencePipeline
from modelflow.core.config import ModelConfig
from modelflow.backends.onnx import OnnxBackend
from modelflow.processors.classify import ClassifyPreprocessor, ClassifyPostprocessor


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

    if backend == "onnxruntime":
        bk = OnnxBackend(model_path, class_list, task_type="classify",
                         half=half, device=device)
    elif backend == "tensorrt":
        from modelflow.backends.tensorrt import TensorrtBackend
        bk = TensorrtBackend(model_path, class_list, task_type="classify",
                             half=half, device=device)
    elif backend == "triton":
        from modelflow.backends.triton import TritonBackend
        bk = TritonBackend(model_path, class_list, task_type="classify",
                           half=half, device=device)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    postprocessor = ClassifyPostprocessor(topk=topk, class_list=class_list)

    return InferencePipeline(preprocessor, bk, postprocessor)


def create_classify_pipeline_from_config(config: ModelConfig) -> InferencePipeline:
    """从 ModelConfig 创建分类 Pipeline"""
    return create_classify_pipeline(
        model_path=config.model_path,
        class_list=config.class_list,
        backend=config.backend,
        processor=config.processor,
        input_size=config.input_size,
        half=config.half,
        device=config.device,
    )
