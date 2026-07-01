# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_export.py
@Author  : zj
@Description: Export module unit tests

Coverage:
- Preprocessing functions in core/utils.py
- Torchvision export via onnx/convert.py
- Ultralytics export via onnx/ultralytics.py (requires network download of models)
- ONNX optimization via onnx/optimize.py

Run:
    python3 -m pytest export/tests/test_export.py -v
    python3 -m pytest export/tests/test_export.py -v -k "preprocess"  # preprocessing tests only
"""

import os
import tempfile
import pytest
import numpy as np

from export._utils import (
    letterbox,
    classify_preprocess,
    detect_preprocess,
    collect_images,
)


class TestPreprocess:
    """Preprocessing function unit tests (no GPU required)"""

    @pytest.fixture
    def dummy_image(self):
        """Create a dummy BGR image (480, 640, 3)"""
        return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_letterbox_square(self, dummy_image):
        """LetterBox resize to square (auto_pad=False, default)"""
        padded, scale, pad = letterbox(dummy_image, target_size=640)
        assert padded.shape[:2] == (640, 640)
        assert scale[0] > 0 and scale[1] > 0
        assert len(pad) == 2

    def test_letterbox_min_rect(self, dummy_image):
        """LetterBox minimum rectangle mode (auto_pad=True, efficient inference)"""
        padded, scale, pad = letterbox(dummy_image, target_size=640, auto_pad=True)
        # Minimum rectangle mode: dimensions should be multiples of stride(32), not necessarily square
        assert padded.shape[0] % 32 == 0
        assert padded.shape[1] % 32 == 0
        assert padded.shape[0] <= 640 and padded.shape[1] <= 640

    def test_letterbox_auto_pad(self, dummy_image):
        """When auto_pad=True, padding should be multiples of 32"""
        padded, scale, pad = letterbox(dummy_image, target_size=640, auto_pad=True)
        assert padded.shape[0] % 32 == 0
        assert padded.shape[1] % 32 == 0

    def test_classify_preprocess_shape(self, dummy_image):
        """Classification preprocessing output shape should be (3, 224, 224)"""
        result = classify_preprocess(dummy_image, crop_size=224)
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_classify_preprocess_normalized(self, dummy_image):
        """After classification preprocessing, data should approximate normal distribution (mean~0, std~1)"""
        result = classify_preprocess(dummy_image, crop_size=224)
        assert -2.0 < result.mean() < 2.0

    def test_detect_preprocess_shape(self, dummy_image):
        """Detection preprocessing uses minimum rectangle mode — dimensions should be multiples of stride(32)"""
        result, scale, pad = detect_preprocess(dummy_image, target_size=640)
        assert result.shape[0] == 3  # CHW
        assert result.shape[1] % 32 == 0
        assert result.shape[2] % 32 == 0
        assert result.dtype == np.float32

    def test_detect_preprocess_range(self, dummy_image):
        """Detection preprocessing output should be in [0, 1] range"""
        result, scale, pad = detect_preprocess(dummy_image, target_size=640)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_collect_images_nonexistent(self):
        """Collecting from a nonexistent directory should return empty list"""
        images = collect_images("/tmp/nonexistent_path_for_test")
        assert images == []


class TestTorchvisionExporter:
    """Torchvision exporter integration tests (requires PyTorch)"""

    def test_load_model(self):
        """Test model loading"""
        from export.onnx.convert import load_torchvision_model
        model = load_torchvision_model("efficientnet_b0")
        import torch.nn as nn
        assert isinstance(model, nn.Module)
        assert model.training is False

    def test_export_onnx(self):
        """Test ONNX export + output comparison"""
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
    """Ultralytics exporter integration tests (requires network)"""

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
    """Validation utility tests"""

    def test_check_onnx_invalid_file(self):
        """Check nonexistent ONNX file"""
        from export._validation import check_onnx
        with pytest.raises(Exception):
            check_onnx("/tmp/nonexistent.onnx")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
