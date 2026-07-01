# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_core.py
@Author  : zj
@Description: 基础设施单元测试（types, config, interfaces）
"""

import pytest
import numpy as np

from modelflow.types import ModelInfo
from modelflow.config import ModelConfig
from modelflow.interfaces import (
    BaseBackend, BasePreprocessor, BasePostprocessor,
    InferencePipeline,
)


class TestModelInfo:
    """ModelInfo 数据结构测试"""

    def test_model_info(self):
        info = ModelInfo(name="input", shape=[1, 3, 640, 640], dtype=np.float32)
        assert info.name == "input"
        assert info.shape == [1, 3, 640, 640]
        assert info.dtype == np.float32
        assert info.index == 0  # default

    def test_model_info_with_index(self):
        info = ModelInfo(name="output", shape=[84, 8400], dtype=np.float32, index=1)
        assert info.index == 1


class TestConfig:
    """配置管理测试"""

    def test_config_defaults(self):
        cfg = ModelConfig(model_path="test.onnx", class_list=["a", "b"])
        assert cfg.model_path == "test.onnx"
        assert cfg.class_list == ["a", "b"]
        assert cfg.backend == "onnxruntime"  # default
        assert cfg.task_type == "detect"      # default
        assert cfg.conf_thres == 0.25
        assert cfg.iou_thres == 0.45

    def test_config_custom_values(self):
        cfg = ModelConfig(
            model_path="yolov8s.onnx",
            class_list=["person", "car"],
            backend="tensorrt",
            task_type="detect",
            half=True,
            conf_thres=0.001,
            iou_thres=0.7,
        )
        assert cfg.half is True
        assert cfg.conf_thres == 0.001


class TestInterfaces:
    """抽象接口测试"""

    def test_inference_pipeline_composition(self):
        """Pipeline 可以组合 mock 组件并执行"""
        class MockPreprocessor:
            def __call__(self, image, **kwargs):
                return np.ones((1, 3, 640, 640), dtype=np.float32)

        class MockBackend:
            def __call__(self, tensor):
                return [np.random.randn(1, 84, 8400).astype(np.float32)]

        class MockPostprocessor:
            def __call__(self, raw, **kwargs):
                return {"boxes": np.array([[10, 20, 100, 200]]),
                        "scores": np.array([0.9]),
                        "class_ids": np.array([0])}

        pipeline = InferencePipeline(MockPreprocessor(), MockBackend(), MockPostprocessor())
        result = pipeline(np.zeros((480, 640, 3), dtype=np.uint8))
        assert "boxes" in result
        assert "scores" in result

    def test_pipeline_infer_raw(self):
        """pipeline.infer() 返回原始输出"""
        class MockBackend:
            def __call__(self, tensor):
                return [np.ones((1, 10))]
        pipeline = InferencePipeline(None, MockBackend(), None)
        raw = pipeline.infer(np.ones((1, 3, 224, 224)))
        assert len(raw) == 1

    def test_pipeline_warmup(self):
        """warmup 调用各组件的 warmup 方法"""
        calls = []
        class MockPre:
            def warmup(self): calls.append("pre")
            def __call__(self, x): return x
        class MockBk:
            def warmup(self): calls.append("bk")
            def __call__(self, x): return [x]
        class MockPost:
            def warmup(self): calls.append("post")
            def __call__(self, x, **kw): return x

        pipeline = InferencePipeline(MockPre(), MockBk(), MockPost())
        pipeline.warmup()
        assert calls == ["pre", "bk", "post"]

    def test_base_backend_interface(self):
        """BaseBackend 子类必须实现 __call__"""
        class ConcreteBackend(BaseBackend):
            def __call__(self, input_data):
                return [input_data]
        bk = ConcreteBackend("test.onnx", ["a", "b"], task_type="detect")
        result = bk(np.ones((1, 3, 640, 640)))
        assert len(result) == 1
        assert bk.model_path == "test.onnx"
        assert bk.task_type == "detect"

    def test_base_backend_print_model_info(self):
        """print_model_info 不抛出异常"""
        class SimpleBackend(BaseBackend):
            def __call__(self, x): return [x]
        bk = SimpleBackend("m.onnx", ["a"])
        bk.print_model_info()  # should not raise

    def test_base_preprocessor_interface(self):
        class MyPre(BasePreprocessor):
            def __call__(self, image, **kwargs):
                return np.ones((1, 3, 224, 224))
        p = MyPre()
        result = p(np.zeros((100, 100, 3)))
        assert result.shape == (1, 3, 224, 224)

    def test_base_postprocessor_interface(self):
        class MyPost(BasePostprocessor):
            def __call__(self, raw, **kwargs):
                return {"class_id": 0}
        p = MyPost()
        assert p([np.array([1, 2, 3])])["class_id"] == 0
