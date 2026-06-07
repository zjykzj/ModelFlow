# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : onnx.py
@Author  : zj
@Description: ONNX Runtime 推理后端

封装 onnxruntime.InferenceSession，提供统一推理接口。
支持 CPU 和 CUDA provider。

用法:
    backend = OnnxBackend("model.onnx", class_list=["person", "car"])
    outputs = backend(input_tensor)
"""

from typing import List, Optional, Dict, Union, Any

import numpy as np
import onnxruntime as ort

from modelflow.core.interfaces import BaseBackend
from modelflow.core.registry import BACKENDS
from modelflow.core.types import ModelInfo
from modelflow.utils.logger import get_logger

logger = get_logger("modelflow.backend.onnx")


@BACKENDS.register("onnxruntime")
class OnnxBackend(BaseBackend):
    """ONNX Runtime 推理后端"""

    def __init__(
        self,
        model_path: str,
        class_list: List[str],
        task_type: Optional[str] = None,
        half: bool = False,
        device: Optional[str] = None,
        providers: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(model_path, class_list, task_type, half, device, **kwargs)

        # 自动选择 provider
        if providers is None:
            if device and "cuda" in device.lower():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self._init_io_info()
        self.warmup()

    def _init_io_info(self):
        """初始化输入输出元信息"""
        inputs = self.session.get_inputs()
        self._input_infos = [
            ModelInfo(name=inp.name, shape=list(inp.shape),
                      dtype=self._to_numpy_dtype(inp.type), index=i)
            for i, inp in enumerate(inputs)
        ]
        self._input_names = [info.name for info in self._input_infos]

        outputs = self.session.get_outputs()
        self._output_infos = [
            ModelInfo(name=out.name, shape=list(out.shape),
                      dtype=self._to_numpy_dtype(out.type), index=i)
            for i, out in enumerate(outputs)
        ]
        self._output_names = [info.name for info in self._output_infos]

        logger.info(f"Model loaded: {self.model_path}")
        logger.info(f"  Inputs:  {[(i.name, i.shape) for i in self._input_infos]}")
        logger.info(f"  Outputs: {[(o.name, o.shape) for o in self._output_infos]}")

    @staticmethod
    def _to_numpy_dtype(onnx_dtype: str) -> np.dtype:
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int8)": np.int8,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
        }
        return dtype_map.get(onnx_dtype, np.float32)

    def _prepare_input(self, input_data: Union[np.ndarray, List, Dict]) -> Dict[str, np.ndarray]:
        if isinstance(input_data, dict):
            return input_data
        if isinstance(input_data, list):
            return dict(zip(self._input_names, input_data))
        if isinstance(input_data, np.ndarray):
            if len(self._input_names) == 1:
                return {self._input_names[0]: input_data}
            raise ValueError(f"Model expects {len(self._input_names)} inputs")
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    def __call__(self, input_data: np.ndarray) -> List[np.ndarray]:
        feed = self._prepare_input(input_data)
        outputs = self.session.run(self._output_names, feed)
        return list(outputs)

    def warmup(self):
        """使用 dummy input 预热模型"""
        try:
            dummy = []
            for info in self._input_infos:
                shape = [1 if d == -1 else d for d in info.shape]
                dummy.append(np.random.randn(*shape).astype(np.float32))
            feed = dict(zip(self._input_names, dummy))
            for _ in range(3):
                self.session.run(self._output_names, feed)
            logger.info("Warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def get_input_info(self) -> ModelInfo:
        return self._input_infos[0]

    def get_output_info(self) -> List[ModelInfo]:
        return self._output_infos
