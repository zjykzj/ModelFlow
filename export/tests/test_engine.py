# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_engine.py
@Author  : zj
@Description: TensorRT engine build and inference tests (requires GPU environment)

Run:
    python3 -m pytest export/tests/test_engine.py -v
    python3 -m pytest export/tests/test_engine.py -v -k "calibrator"
"""

import os
import tempfile
import pytest
import numpy as np

# ==================== Global Skip Conditions ====================

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

requires_gpu = pytest.mark.skipif(
    not (HAS_TRT and (HAS_TORCH_CUDA or HAS_PYCUDA)),
    reason="Requires GPU with TensorRT and either PyTorch CUDA or PyCUDA",
)


class TestCalibrator:
    """INT8 calibrator unit tests (file scanning, data self-healing, shape validation)"""

    @pytest.fixture
    def calib_dir(self):
        """Create a temporary directory containing dummy calibration data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 5 valid .bin files
            for i in range(5):
                data = np.random.randn(3 * 640 * 640).astype(np.float32)
                data.tofile(os.path.join(tmpdir, f"img_{i:04d}.bin"))
            # Create one with mismatched size (anomalous file)
            bad_data = np.random.randn(100).astype(np.float32)
            bad_data.tofile(os.path.join(tmpdir, "bad_img.bin"))
            yield tmpdir

    @requires_gpu
    def test_torch_calibrator_init(self, calib_dir):
        """TorchCalibrator initialization"""
        from export.tensorrt.calibrator import TorchCalibrator

        calib = TorchCalibrator(
            calib_data_dir=calib_dir,
            input_shape=(1, 3, 640, 640),
            cache_file="/tmp/test_calib_torch.cache",
        )
        assert calib.total == 5  # Bad files are skipped during get_batch
        assert calib.get_batch_size() == 1

    @requires_gpu
    def test_torch_calibrator_get_batch(self, calib_dir):
        """TorchCalibrator get_batch"""
        from export.tensorrt.calibrator import TorchCalibrator

        calib = TorchCalibrator(
            calib_data_dir=calib_dir,
            input_shape=(1, 3, 640, 640),
        )
        ptr = calib.get_batch(None)
        assert ptr is not None
        assert isinstance(ptr, list)
        assert len(ptr) == 1
        assert isinstance(ptr[0], int)

    @pytest.mark.skipif(not HAS_PYCUDA, reason="Requires pycuda")
    def test_pycuda_calibrator_init(self, calib_dir):
        """PyCudaCalibrator initialization"""
        from export.tensorrt.calibrator import PyCudaCalibrator

        calib = PyCudaCalibrator(
            calib_data_dir=calib_dir,
            input_shape=(1, 3, 640, 640),
            cache_file="/tmp/test_calib_pycuda.cache",
        )
        assert calib.total == 5

    @pytest.mark.skipif(not HAS_TRT, reason="Requires TensorRT")
    def test_calibrator_empty_dir(self):
        """Empty directory should raise an exception"""
        from export.tensorrt.calibrator import BaseCalibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                BaseCalibrator(
                    calib_data_dir=tmpdir,
                    input_shape=(1, 3, 640, 640),
                )

    @pytest.mark.skipif(not HAS_TRT, reason="Requires TensorRT")
    def test_calibrator_nonexistent_dir(self):
        """Nonexistent directory should raise an exception"""
        from export.tensorrt.calibrator import BaseCalibrator

        with pytest.raises(FileNotFoundError):
            BaseCalibrator(
                calib_data_dir="/tmp/nonexistent_calib",
                input_shape=(1, 3, 640, 640),
            )


class TestTRTBuild:
    """TensorRT engine build tests (requires GPU + TensorRT)"""

    @requires_gpu
    def test_build_fp16_invalid_onnx(self):
        """Invalid ONNX file should return False"""
        from export.tensorrt.build_fp16 import build_fp16_engine

        result = build_fp16_engine(
            onnx_path="/tmp/nonexistent.onnx",
            output_path="/tmp/nonexistent.engine",
            use_trtexec=False,
        )
        assert result is False

    @requires_gpu
    def test_triton_config_generation(self):
        """Triton configuration generation"""
        from export.triton import TritonConfigGenerator

        gen = TritonConfigGenerator(
            model_name="Test_Detect",
            backend="tensorrt",
            task="detect",
        )
        config = gen.generate()
        assert 'name: "Test_Detect"' in config
        assert 'platform: "tensorrt_plan"' in config
        assert "dims: [84, 8400]" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
