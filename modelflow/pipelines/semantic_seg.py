# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : semantic_seg.py
@Author  : zj
@Description: 语义分割 Pipeline 工厂
"""

from typing import List, Optional, Tuple

from modelflow.core.interfaces import InferencePipeline
from modelflow.processors.semantic_seg import (
    SemanticSegPreprocessor,
    SemanticSegPostprocessor,
)


def create_semantic_seg_pipeline(
    model_path: str,
    class_list: List[str],
    backend: str = "onnxruntime",
    input_size: Optional[Tuple[int, int]] = None,
    half: bool = False,
    device: Optional[str] = None,
    apply_colormap: bool = True,
) -> InferencePipeline:
    """创建语义分割 Pipeline

    Args:
        model_path: 模型路径
        class_list: 类别名称列表
        backend: 后端类型
        input_size: 输入尺寸 (H, W)，None 表示保持原图
        half: 是否使用半精度
        device: 设备
        apply_colormap: 是否生成彩色掩码

    Returns:
        InferencePipeline 实例
    """
    preprocessor = SemanticSegPreprocessor(target_size=input_size)

    if backend == "onnxruntime":
        from modelflow.backends.onnx import OnnxBackend
        bk = OnnxBackend(model_path, class_list, task_type="semantic_seg",
                         half=half, device=device)
    elif backend == "tensorrt":
        from modelflow.backends.tensorrt import TensorrtBackend
        bk = TensorrtBackend(model_path, class_list, task_type="semantic_seg",
                             half=half, device=device)
    elif backend == "triton":
        from modelflow.backends.triton import TritonBackend
        bk = TritonBackend(model_path, class_list, task_type="semantic_seg",
                           half=half, device=device)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    postprocessor = SemanticSegPostprocessor(
        apply_colormap=apply_colormap,
        num_classes=len(class_list),
    )

    return InferencePipeline(preprocessor, bk, postprocessor)
