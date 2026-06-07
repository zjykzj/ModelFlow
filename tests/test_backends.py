# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_backends.py
@Author  : zj
@Description: 推理后端测试（无真实模型，测试初始化逻辑）
"""

import pytest
import numpy as np

from modelflow.core.registry import BACKENDS

# ==================== 环境检测 ====================

try:
    import tensorrt as trt  # noqa: F401
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

requires_trt = pytest.mark.skipif(not HAS_TRT, reason="TensorRT not installed")


class TestOnnxBackend:
    """ONNX 后端测试（无需 GPU）"""

    def test_import(self):
        from modelflow.backends.onnx import OnnxBackend
        assert OnnxBackend is not None

    def test_instantiation_fails_with_nonexistent_model(self):
        from modelflow.backends.onnx import OnnxBackend
        with pytest.raises(Exception):
            OnnxBackend("/tmp/nonexistent_model.onnx", class_list=["a"])

    def test_registered(self):
        assert "onnxruntime" in BACKENDS


class TestTensorrtBackend:
    """TensorRT 后端测试"""

    @pytest.mark.skipif(not HAS_TRT, reason="TensorRT not installed")
    def test_import(self):
        from modelflow.backends.tensorrt import TensorrtBackend
        assert TensorrtBackend is not None

    @requires_trt
    def test_instantiation_fails_with_nonexistent_engine(self):
        from modelflow.backends.tensorrt import TensorrtBackend
        with pytest.raises(Exception):
            TensorrtBackend("/tmp/nonexistent.engine", class_list=["a"])

    @requires_trt
    def test_registered(self):
        assert "tensorrt" in BACKENDS


class TestTritonBackend:
    """Triton 后端测试"""

    def test_import(self):
        from modelflow.backends.triton import TritonBackend
        assert TritonBackend is not None

    def test_instantiation_fails_without_server(self):
        from modelflow.backends.triton import TritonBackend
        with pytest.raises(Exception):
            TritonBackend("test_model", class_list=["a"],
                          server_url="localhost:19999", timeout=1)

    def test_registered(self):
        assert "triton" in BACKENDS


class TestBackendInterface:
    """后端接口一致性测试"""

    def test_backends_registered_onnxruntime(self):
        assert "onnxruntime" in BACKENDS

    def test_backend_base_class_available(self):
        from modelflow.backends.base import BaseBackend
        from modelflow.core.interfaces import BaseBackend as BaseIface
        assert BaseBackend is BaseIface
