# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_core.py
@Author  : zj
@Description: core/ 基础设施单元测试

覆盖: Registry, types, config, BaseExporter 接口
"""

import pytest
import numpy as np

from modelflow.core.registry import Registry, BACKENDS, PROCESSORS, DATASETS, METRICS, EVALUATORS
from modelflow.core.types import TaskType, BackendType, ProcessorType, ModelInfo
from modelflow.core.config import ModelConfig
from modelflow.core.interfaces import (
    BaseBackend, BasePreprocessor, BasePostprocessor,
    BaseDataset, BaseMetrics, BaseEvaluator, BaseVisualizer,
    InferencePipeline,
)


class TestRegistry:
    """注册机制测试"""

    def test_register_and_get(self):
        reg = Registry("test")
        @reg.register("my_component")
        class MyComp:
            pass
        assert reg.get("my_component") is MyComp

    def test_register_default_name(self):
        reg = Registry("test")
        class MyComp:
            pass
        reg.register()(MyComp)
        assert reg.get("mycomp") is MyComp

    def test_register_overwrite_warning(self):
        reg = Registry("test")
        class A: pass
        class B: pass
        reg.register("x")(A)
        # Second registration should warn but succeed
        reg.register("x")(B)
        assert reg.get("x") is B

    def test_build(self):
        reg = Registry("test")
        @reg.register("mycomp")
        class MyComp:
            def __init__(self, a=1):
                self.a = a
        obj = reg.build("mycomp", a=42)
        assert obj.a == 42

    def test_get_nonexistent(self):
        reg = Registry("test")
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list(self):
        reg = Registry("test")
        reg.register("a")(type("A", (), {}))
        reg.register("b")(type("B", (), {}))
        assert reg.list() == ["a", "b"]

    def test_contains(self):
        reg = Registry("test")
        reg.register("x")(type("X", (), {}))
        assert "x" in reg
        assert "y" not in reg

    def test_global_registries_exist(self):
        assert BACKENDS.list() is not None
        assert PROCESSORS.list() is not None
        assert DATASETS.list() is not None
        assert METRICS.list() is not None
        assert EVALUATORS.list() is not None


class TestTypes:
    """类型枚举和数据结构测试"""

    def test_task_type_values(self):
        assert TaskType.CLASSIFY.value == "classify"
        assert TaskType.DETECT.value == "detect"
        assert TaskType.INSTANCE_SEGMENT.value == "instance_segment"
        assert len(TaskType) >= 5

    def test_backend_type_values(self):
        assert BackendType.ONNXRUNTIME.value == "onnxruntime"
        assert BackendType.TENSORRT.value == "tensorrt"

    def test_processor_type_values(self):
        assert ProcessorType.NUMPY.value == "numpy"
        assert ProcessorType.TORCH.value == "torch"

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

    def test_config_to_dict(self):
        cfg = ModelConfig(model_path="m.onnx", class_list=["x"], task_type="classify")
        d = cfg.to_dict()
        assert d["model_path"] == "m.onnx"
        assert d["task_type"] == "classify"

    def test_config_from_dict(self):
        cfg = ModelConfig.from_dict({
            "model_path": "m.onnx",
            "class_list": ["a"],
            "task_type": "detect",
            "backend": "tensorrt",
        })
        assert cfg.backend == "tensorrt"

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

    def test_base_dataset_interface(self):
        """BaseDataset 抽象方法"""
        class MyDataset(BaseDataset):
            def __len__(self): return 10
            def __getitem__(self, idx):
                return (np.zeros((100, 100, 3)), {"id": idx})
            def get_gt_json(self):
                return ""
        ds = MyDataset()
        assert len(ds) == 10
        img, gt = ds[0]
        assert gt["id"] == 0

    def test_base_metrics_interface(self):
        class MyMetrics(BaseMetrics):
            def update(self, p, g): pass
            def compute(self): return {"acc": 0.5}
            def reset(self): pass
        m = MyMetrics()
        assert m.compute()["acc"] == 0.5

    def test_base_evaluator_interface(self):
        class MyEval(BaseEvaluator):
            def run(self): return {"mAP": 0.5}
        ev = MyEval(None, None)
        assert ev.run()["mAP"] == 0.5

    def test_base_visualizer_interface(self):
        class MyViz(BaseVisualizer):
            def draw(self, img, pred, **kw):
                return img
        v = MyViz()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = v.draw(img, {})
        assert result.shape == (100, 100, 3)
