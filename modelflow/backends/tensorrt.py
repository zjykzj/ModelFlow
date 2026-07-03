# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : tensorrt.py
@Author  : zj
@Description: TensorRT 推理后端

封装 TensorRT Python API，管理 CUDA buffer 的分配和释放。

用法:
    backend = TensorrtBackend("model.engine", class_list=["person", "car"])
    outputs = backend(input_tensor)
"""

from typing import List, Optional

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 (must import to init CUDA context)

from modelflow.interfaces import BaseBackend
from modelflow.types import ModelInfo
from utils.logger import get_logger

logger = get_logger("modelflow.backend.tensorrt")


class TensorrtBackend(BaseBackend):
    """TensorRT 推理后端"""

    def __init__(
        self,
        model_path: str,
        class_list: List[str],
        task_type: Optional[str] = None,
        half: bool = False,
        device: Optional[str] = None,
        max_batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(model_path, class_list, task_type, half, device, **kwargs)
        self.max_batch_size = max_batch_size

        # 加载引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        try:
            with open(self.model_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(
                f"TensorRT engine deserialization failed for {self.model_path}: {e}"
            ) from e
        self.context = self.engine.create_execution_context()

        self._init_io_info()
        self._allocate_buffers()
        self.warmup()

    def _init_io_info(self):
        self._input_infos = []
        self._output_infos = []
        self._input_names = []
        self._output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = list(self.engine.get_tensor_shape(name))
            info = ModelInfo(name=name, shape=shape, dtype=dtype, index=i)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_infos.append(info)
                self._input_names.append(name)
            else:
                self._output_infos.append(info)
                self._output_names.append(name)

        logger.info(f"Engine loaded: {self.model_path}")
        logger.info(f"  Inputs:  {[(i.name, i.shape) for i in self._input_infos]}")
        logger.info(f"  Outputs: {[(o.name, o.shape) for o in self._output_infos]}")

    def _allocate_buffers(self):
        """分配 host 和 device buffer"""
        self._device_buffers = {}
        self._host_buffers = {}

        for info in self._input_infos + self._output_infos:
            size = int(np.prod([1 if d == -1 else d for d in info.shape]))
            self._host_buffers[info.name] = cuda.pagelocked_empty(size, dtype=info.dtype)
            self._device_buffers[info.name] = cuda.mem_alloc(self._host_buffers[info.name].nbytes)

    def __call__(self, input_data: np.ndarray) -> List[np.ndarray]:
        # H2D
        np.copyto(self._host_buffers[self._input_names[0]], input_data.ravel())
        cuda.memcpy_htod(self._device_buffers[self._input_names[0]],
                         self._host_buffers[self._input_names[0]])

        # 设置 tensor 地址并推理
        for name in self._input_names:
            self.context.set_tensor_address(name, int(self._device_buffers[name]))
        for name in self._output_names:
            self.context.set_tensor_address(name, int(self._device_buffers[name]))

        self.context.execute_async_v3(stream_handle=0)

        # D2H
        outputs = []
        for name in self._output_names:
            cuda.memcpy_dtoh(self._host_buffers[name],
                             self._device_buffers[name])
            out = self._host_buffers[name].copy()
            # 恢复原始 shape
            info = self._output_infos[self._output_names.index(name)]
            shape = [1 if d == -1 else d for d in info.shape]
            outputs.append(out.reshape(shape))

        if len(outputs) != len(self._output_names):
            raise ValueError(
                f"Output tensor count mismatch: expected {len(self._output_names)}, "
                f"got {len(outputs)}"
            )
        return outputs

    def warmup(self):
        try:
            dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
            self.__call__(dummy)
            logger.info("Warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def get_input_info(self) -> ModelInfo:
        return self._input_infos[0]

    def get_output_info(self) -> List[ModelInfo]:
        return self._output_infos

    def __del__(self):
        if hasattr(self, "_device_buffers"):
            for buf in self._device_buffers.values():
                try:
                    buf.free()
                except Exception:
                    pass  # CUDA context may already be destroyed at GC time
        if hasattr(self, "_host_buffers"):
            self._host_buffers.clear()
