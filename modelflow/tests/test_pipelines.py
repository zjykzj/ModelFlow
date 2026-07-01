# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_pipelines.py
@Author  : zj
@Description: Pipeline 工厂测试

注意: Pipeline 工厂会实际创建 backend（OnnxBackend 需要真实 .onnx 文件），
因此测试前置条件和后处理器的构建，不测试 backend 初始化（需真实模型）。
"""

import pytest
import os
import tempfile

from modelflow.pipelines import (
    create_classify_pipeline,
    create_detect_pipeline,
    create_segment_pipeline,
    create_semantic_seg_pipeline,
)

_CLASSES = ["cat", "dog", "bird"]

# 检测 TensorRT 是否可用
try:
    import tensorrt  # noqa: F401
    HAS_TRT = True
except ImportError:
    HAS_TRT = False


class TestClassifyPipeline:
    """分类 Pipeline 工厂测试"""

    def test_create_unsupported_backend(self):
        with pytest.raises(ValueError):
            create_classify_pipeline(
                model_path="test.onnx", class_list=_CLASSES,
                backend="invalid_backend",
            )


class TestDetectPipeline:
    """检测 Pipeline 工厂测试"""

    def test_create_unsupported_backend(self):
        with pytest.raises(ValueError):
            create_detect_pipeline(
                model_path="test.onnx", class_list=_CLASSES,
                backend="invalid_backend",
            )


class TestSegmentPipeline:
    """分割 Pipeline 工厂测试"""

    def test_create_unsupported_backend(self):
        with pytest.raises(ValueError):
            create_segment_pipeline(
                model_path="test.onnx", class_list=_CLASSES,
                backend="invalid_backend",
            )


class TestSemanticSegPipeline:
    """语义分割 Pipeline 工厂测试"""

    def test_create_unsupported_backend(self):
        with pytest.raises(ValueError):
            create_semantic_seg_pipeline(
                model_path="test.onnx", class_list=_CLASSES,
                backend="invalid_backend",
            )
