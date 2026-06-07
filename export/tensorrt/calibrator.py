# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : calibrator.py
@Author  : zj
@Description: TensorRT INT8 量化校准器

分层设计：
    BaseCalibrator      — 通用逻辑（文件扫描、数据自愈、日志、缓存）
    ├── TorchCalibrator   — PyTorch 数据传输（服务器环境）
    └── PyCudaCalibrator  — PyCUDA 数据传输（Jetson/嵌入式环境）
"""

import os
from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import tensorrt as trt


class BaseCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 校准器基类

    处理文件扫描、数据自愈、进度日志和缓存读写等通用逻辑。
    子类只需实现数据传输部分（host→device）。
    """

    def __init__(
        self,
        calib_data_dir: str,
        input_shape: Tuple[int, int, int, int],
        cache_file: str = "calib.cache",
        max_calib_size: Optional[int] = None,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.cache_file = cache_file

        if not os.path.isdir(calib_data_dir):
            raise FileNotFoundError(f"Calibration directory not found: {calib_data_dir}")

        # 扫描 .bin 文件
        self.files: List[str] = sorted([
            os.path.join(calib_data_dir, f)
            for f in os.listdir(calib_data_dir)
            if f.endswith(".bin")
        ])
        if not self.files:
            raise FileNotFoundError(f"No .bin files found in {calib_data_dir}")

        if max_calib_size is not None and max_calib_size < len(self.files):
            self.files = self.files[:max_calib_size]

        # 解析输入 shape
        if len(input_shape) != 4:
            raise ValueError(f"input_shape must be (N, C, H, W), got {input_shape}")
        self.n, self.c, self.h, self.w = input_shape
        self.single_vol = self.c * self.h * self.w

        self.idx = 0
        self.total = len(self.files)

        print(f"[Calibrator] Loaded {self.total} calibration files")
        print(f"[Calibrator] Input shape: {input_shape}")
        print(f"[Calibrator] Per-image volume: {self.single_vol} floats "
              f"({self.single_vol * 4 / 1024 / 1024:.2f} MB)")

    def get_batch_size(self) -> int:
        return 1

    @abstractmethod
    def _host_to_device(self, data: np.ndarray) -> int:
        """子类实现：将 host 数据拷贝到 device，返回 device 指针"""
        ...

    def get_batch(self, names) -> Optional[List[int]]:
        while self.idx < self.total:
            file_path = self.files[self.idx]
            self.idx += 1
            try:
                data = np.fromfile(file_path, dtype=np.float32)
                # 尺寸校验
                if data.size != self.single_vol:
                    print(f"  ⚠️  Skip {os.path.basename(file_path)}: size mismatch "
                          f"({data.size} != {self.single_vol})")
                    continue
                # 数据自愈：修复 NaN/Inf
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    print(f"  ⚠️  Fix {os.path.basename(file_path)}: NaN/Inf detected")
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                d_ptr = self._host_to_device(data)

                if self.idx % max(1, self.total // 10) == 0 or self.idx == self.total:
                    print(f"  🔄 Progress: {self.idx} / {self.total}")

                return [d_ptr]
            except Exception as e:
                print(f"  ❌ Read failed {os.path.basename(file_path)}: {e}")
                continue

        return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[Calibrator] ✅ Cache saved to {self.cache_file}")


class TorchCalibrator(BaseCalibrator):
    """基于 PyTorch 的 INT8 校准器

    适用：RTX 服务器、AutoDL、已有 PyTorch 的环境
    优势：无需额外安装 pycuda，开发调试便捷
    """

    def __init__(self, *args, **kwargs):
        import torch

        super().__init__(*args, **kwargs)
        self.host_input = torch.empty(self.single_vol, dtype=torch.float32, pin_memory=True)
        self.device_input = torch.empty(self.single_vol, dtype=torch.float32, device="cuda")

    def _host_to_device(self, data: np.ndarray) -> int:
        import torch
        self.host_input[:] = torch.from_numpy(data)
        self.device_input.copy_(self.host_input, non_blocking=False)
        return int(self.device_input.data_ptr())


class PyCudaCalibrator(BaseCalibrator):
    """基于 PyCUDA 的 INT8 校准器

    适用：NVIDIA Jetson、嵌入式设备、Docker 精简镜像
    优势：极致轻量，无重型框架依赖
    """

    def __init__(self, *args, **kwargs):
        import pycuda.driver as cuda

        super().__init__(*args, **kwargs)
        self.host_input = cuda.pagelocked_empty(self.single_vol, dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.host_input.nbytes)

    def _host_to_device(self, data: np.ndarray) -> int:
        import pycuda.driver as cuda

        np.copyto(self.host_input, data)
        cuda.memcpy_htod(self.device_input, self.host_input)
        return int(self.device_input)
