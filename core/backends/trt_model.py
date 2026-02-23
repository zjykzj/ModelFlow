# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/23 14:27
@File    : trt_model.py
@Author  : zj
@Description: 
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)


@dataclass
class IOInfo:
    """输入/输出信息数据类"""
    name: str
    shape: List[int]
    dtype: np.dtype
    index: int


class TRTModel:
    """
    统一的TensorRT模型推理类
    支持分类、检测、分割三种任务类型
    接口设计与ONNXModel保持一致
    """

    def __init__(
            self,
            engine_path: str,
            class_list: List[str],
            label_list: Optional[List[str]] = None,
            half: bool = False,
            device: Optional[str] = None,
            providers: Optional[List[str]] = None,  # 兼容ONNXModel接口（TRT不使用）
            stride: int = 32,
            max_batch_size: int = 1,
    ):
        """
        初始化TensorRT模型推理类
        """
        self.engine_path = engine_path
        self.class_list = class_list
        self.label_list = label_list
        self.half = half
        self.device = device if device is not None else 0
        self.stride = stride
        self.max_batch_size = max_batch_size

        logger.info(f"engine_path: {engine_path}")
        logger.info(f"half: {half}")

        # 加载Engine
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        logger.info(f"Engine loaded successfully")

        # 获取输入输出完整信息
        self._init_io_info()

        # 分配缓冲区
        self._allocate_buffers()

        # 创建CUDA流
        self.stream = cuda.Stream()

        # 热身
        self.__warmup()

    def _load_engine(self) -> trt.ICudaEngine:
        """加载TensorRT Engine"""
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to load engine from {self.engine_path}")
            return engine

    def _init_io_info(self):
        """初始化输入输出信息（包含name, shape, dtype）"""
        # 获取输入信息
        self.input_infos: List[IOInfo] = []
        self.input_names: List[str] = []
        self.input_shapes: List[List[int]] = []
        self.input_dtypes: List[np.dtype] = []
        self.input_bindings: List[int] = []

        # 获取输出信息
        self.output_infos: List[IOInfo] = []
        self.output_names: List[str] = []
        self.output_shapes: List[List[int]] = []
        self.output_dtypes: List[np.dtype] = []
        self.output_bindings: List[int] = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = list(self.engine.get_binding_shape(i))
            trt_dtype = self.engine.get_binding_dtype(i)
            # 修复：确保转换为 numpy.dtype 对象
            dtype = np.dtype(trt.nptype(trt_dtype))
            is_input = self.engine.binding_is_input(i)

            io_info = IOInfo(
                name=name,
                shape=shape,
                dtype=dtype,
                index=i
            )

            if is_input:
                self.input_infos.append(io_info)
                self.input_names.append(name)
                self.input_shapes.append(shape)
                self.input_dtypes.append(dtype)
                self.input_bindings.append(i)
            else:
                self.output_infos.append(io_info)
                self.output_names.append(name)
                self.output_shapes.append(shape)
                self.output_dtypes.append(dtype)
                self.output_bindings.append(i)

            logger.info(f"Binding[{i}]: name='{name}', shape={shape}, dtype={dtype}, is_input={is_input}")

        logger.info(f"input_names: {self.input_names}")
        logger.info(f"input_shapes: {self.input_shapes}")
        logger.info(f"input_dtypes: {self.input_dtypes}")
        logger.info(f"output_names: {self.output_names}")
        logger.info(f"output_shapes: {self.output_shapes}")
        logger.info(f"output_dtypes: {self.output_dtypes}")

    def _allocate_buffers(self):
        """分配GPU显存和主机pinned内存"""
        self.d_inputs = []
        self.d_outputs = []
        self.h_inputs = []
        self.h_outputs = []

        # 输入缓冲区
        for i, (shape, dtype) in enumerate(zip(self.input_shapes, self.input_dtypes)):
            volume = trt.volume(shape)
            # 修复：确保 dtype 是 numpy.dtype 对象
            np_dtype = np.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
            nbytes = volume * np_dtype.itemsize
            self.d_inputs.append(cuda.mem_alloc(nbytes))
            self.h_inputs.append(cuda.pagelocked_empty(volume, dtype=np_dtype))

        # 输出缓冲区
        for i, (shape, dtype) in enumerate(zip(self.output_shapes, self.output_dtypes)):
            volume = trt.volume(shape)
            # 修复：确保 dtype 是 numpy.dtype 对象
            np_dtype = np.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
            nbytes = volume * np_dtype.itemsize
            self.d_outputs.append(cuda.mem_alloc(nbytes))
            self.h_outputs.append(cuda.pagelocked_empty(volume, dtype=np_dtype))

        logger.info("Buffers allocated successfully")

    def __warmup(self):
        """模型热身"""
        try:
            dummy_inputs = []
            for shape in self.input_shapes:
                processed_shape = [1 if dim == -1 else dim for dim in shape]
                dummy_input = np.random.randn(*processed_shape).astype(np.float32)
                dummy_inputs.append(dummy_input)

            for _ in range(3):
                self.infer(dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs)

            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _prepare_input(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """准备输入数据，支持多种输入格式"""
        if isinstance(input_data, dict):
            input_list = []
            for name in self.input_names:
                if name not in input_data:
                    raise ValueError(f"Missing input: {name}")
                input_list.append(input_data[name])
            return input_list
        elif isinstance(input_data, list):
            if len(input_data) != len(self.input_names):
                raise ValueError(
                    f"Input list length ({len(input_data)}) doesn't match "
                    f"model input count ({len(self.input_names)})"
                )
            return input_data
        elif isinstance(input_data, np.ndarray):
            if len(self.input_names) == 1:
                return [input_data]
            else:
                raise ValueError(
                    f"Model expects {len(self.input_names)} inputs, "
                    f"but got single array"
                )
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def __call__(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
            output_names: Optional[List[str]] = None,
            return_dict: bool = False
    ) -> Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray], np.ndarray]:
        """执行模型推理"""
        input_list = self._prepare_input(input_data)
        outputs = self.infer(input_list)

        if return_dict:
            if output_names is None:
                output_names = self.output_names
            result = {}
            for name, output in zip(self.output_names, outputs):
                if name in output_names:
                    result[name] = output
            return result
        else:
            return outputs if len(outputs) > 1 else outputs[0]

    def infer(
            self,
            input_data: Union[np.ndarray, List[np.ndarray]]
    ) -> List[np.ndarray]:
        """核心推理方法"""
        if isinstance(input_data, np.ndarray):
            input_data = [input_data]

        assert len(input_data) == len(self.input_shapes), \
            f"Input count mismatch: {len(input_data)} vs {len(self.input_shapes)}"

        # 将数据复制到 pinned memory
        for i, (h_inp, inp) in enumerate(zip(self.h_inputs, input_data)):
            expected_shape = self.input_shapes[i]
            processed_shape = tuple(
                inp.shape[j] if expected_shape[j] == -1 else expected_shape[j]
                for j in range(len(expected_shape))
            )

            assert inp.shape == processed_shape or inp.size == trt.volume(processed_shape), \
                f"Input {i} shape mismatch: {inp.shape} vs {processed_shape}"

            np.copyto(h_inp, inp.ravel())

        # H2D
        for h_inp, d_inp in zip(self.h_inputs, self.d_inputs):
            cuda.memcpy_htod_async(d_inp, h_inp, self.stream)

        bindings = [None] * self.engine.num_bindings
        for i, binding_idx in enumerate(self.input_bindings):
            bindings[binding_idx] = int(self.d_inputs[i])
        for i, binding_idx in enumerate(self.output_bindings):
            bindings[binding_idx] = int(self.d_outputs[i])

        # 推理
        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=self.stream.handle
        )

        # D2H
        for h_out, d_out in zip(self.h_outputs, self.d_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
        # 同步
        self.stream.synchronize()

        # reshape 输出
        outputs = [
            h_out.reshape(shape)
            for h_out, shape in zip(self.h_outputs, self.output_shapes)
        ]

        return outputs

    def forward(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
            **kwargs
    ) -> Any:
        """前向传播"""
        return self.__call__(input_data, **kwargs)

    def get_input_info(self, index: int = 0) -> IOInfo:
        """获取指定输入的信息"""
        return self.input_infos[index]

    def get_output_info(self, index: int = 0) -> IOInfo:
        """获取指定输出的信息"""
        return self.output_infos[index]

    def print_model_info(self):
        """打印模型完整信息"""
        print("=" * 60)
        print(f"Engine Path: {self.engine_path}")
        print("=" * 60)
        print("INPUTS:")
        for i, info in enumerate(self.input_infos):
            print(f"  [{i}] name={info.name}, shape={info.shape}, dtype={info.dtype}")
        print("OUTPUTS:")
        for i, info in enumerate(self.output_infos):
            print(f"  [{i}] name={info.name}, shape={info.shape}, dtype={info.dtype}")
        print("=" * 60)

    def __del__(self):
        """析构函数，释放资源"""
        try:
            if hasattr(self, 'context') and self.context:
                del self.context
            if hasattr(self, 'engine') and self.engine:
                del self.engine
        except:
            pass
