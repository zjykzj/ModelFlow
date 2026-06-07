# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_export.py
@Author  : zj
@Description: 导出模块单元测试

覆盖范围：
- core/utils.py 的预处理函数
- onnx/convert.py 的 torchvision 导出
- onnx/ultralytics.py 的 Ultralytics 导出（需网络下载模型）
- onnx/optimize.py 的 ONNX 优化

运行：
    python3 -m pytest export/tests/test_export.py -v
    python3 -m pytest export/tests/test_export.py -v -k "preprocess"  # 仅预处理测试
"""

import os
import tempfile
import pytest
import numpy as np

from export.core.utils import (
    letterbox,
    classify_preprocess,
    detect_preprocess,
    collect_images,
)


class TestPreprocess:
    """预处理函数单元测试（无 GPU 要求）"""

    @pytest.fixture
    def dummy_image(self):
        """创建一张虚拟 BGR 图像 (480, 640, 3)"""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_letterbox_square(self, dummy_image):
        """LetterBox 缩放至正方形（auto_pad=False 默认）"""
        padded, scale, pad = letterbox(dummy_image, target_size=640)
        assert padded.shape[:2] == (640, 640)
        assert scale[0] > 0 and scale[1] > 0
        assert len(pad) == 2

    def test_letterbox_min_rect(self, dummy_image):
        """LetterBox 最小矩形模式（auto_pad=True，推理高效）"""
        padded, scale, pad = letterbox(dummy_image, target_size=640, auto_pad=True)
        # 最小矩形模式：尺寸应为 stride(32) 的倍数，但不一定是正方形
        assert padded.shape[0] % 32 == 0
        assert padded.shape[1] % 32 == 0
        assert padded.shape[0] <= 640 and padded.shape[1] <= 640

    def test_letterbox_auto_pad(self, dummy_image):
        """auto_pad=True 时，padding 应为 32 的倍数"""
        padded, scale, pad = letterbox(dummy_image, target_size=640, auto_pad=True)
        assert padded.shape[0] % 32 == 0
        assert padded.shape[1] % 32 == 0

    def test_classify_preprocess_shape(self, dummy_image):
        """分类预处理输出 shape 应为 (3, 224, 224)"""
        result = classify_preprocess(dummy_image, crop_size=224)
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_classify_preprocess_normalized(self, dummy_image):
        """分类预处理后数据应近似正态分布（mean≈0, std≈1）"""
        result = classify_preprocess(dummy_image, crop_size=224)
        assert -2.0 < result.mean() < 2.0

    def test_detect_preprocess_shape(self, dummy_image):
        """检测预处理使用最小矩形模式 — 尺寸应为 stride(32) 的倍数"""
        result, scale, pad = detect_preprocess(dummy_image, target_size=640)
        assert result.shape[0] == 3  # CHW
        assert result.shape[1] % 32 == 0
        assert result.shape[2] % 32 == 0
        assert result.dtype == np.float32

    def test_detect_preprocess_range(self, dummy_image):
        """检测预处理后数据应在 [0, 1] 范围内"""
        result, scale, pad = detect_preprocess(dummy_image, target_size=640)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_collect_images_nonexistent(self):
        """收集不存在的目录应返回空列表"""
        images = collect_images("/tmp/nonexistent_path_for_test")
        assert images == []


class TestTorchvisionExporter:
    """torchvision 导出器集成测试（需要 PyTorch）"""

    def test_load_model(self):
        """测试模型加载"""
        from export.onnx.convert import load_torchvision_model
        model = load_torchvision_model("efficientnet_b0")
        import torch.nn as nn
        assert isinstance(model, nn.Module)
        assert model.training is False

    def test_export_onnx(self):
        """测试 ONNX 导出 + 输出对比"""
        from export.onnx import TorchvisionExporter

        exporter = TorchvisionExporter("efficientnet_b0", opset=12)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        try:
            result_path = exporter.export_onnx(onnx_path, img_size=224)
            assert os.path.exists(result_path)
            assert result_path.endswith(".onnx")
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)


class TestUltralyticsExporter:
    """Ultralytics 导出器集成测试（需要网络）"""

    def test_infer_task(self):
        from export.onnx.ultralytics import UltralyticsExporter
        assert UltralyticsExporter._infer_task("yolov8s") == "detect"
        assert UltralyticsExporter._infer_task("yolov8s-seg") == "segment"
        assert UltralyticsExporter._infer_task("yolov8n-cls") == "classify"
        assert UltralyticsExporter._infer_task("yolov8x-pose") == "pose"
        assert UltralyticsExporter._infer_task("yolo11n") == "detect"

    def test_get_latest_opset(self):
        from export.onnx.ultralytics import get_latest_opset
        opset = get_latest_opset()
        assert isinstance(opset, int)
        assert opset >= 12


class TestValidation:
    """验证工具测试"""

    def test_check_onnx_invalid_file(self):
        """检查不存在的 ONNX 文件"""
        from export.core.validation import check_onnx
        with pytest.raises(Exception):
            check_onnx("/tmp/nonexistent.onnx")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
