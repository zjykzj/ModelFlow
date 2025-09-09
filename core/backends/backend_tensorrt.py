# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/7 16:26
@File    : backend_tensorrt.py
@Author  : zj
@Description:

# Start Docker Container
>>>docker run -it --gpus=all --shm-size=16g -v /etc/localtime:/etc/localtime -v $(pwd):/workdir --workdir=/workdir --name tensorrt-v7.x nvcr.io/nvidia/pytorch:20.12-py3
>>>docker run -it --gpus=all --shm-size=16g -v /etc/localtime:/etc/localtime -v $(pwd):/workdir --workdir=/workdir --name tensorrt-v8.x ultralytics/yolov5:v7.0

# Convert onnx to engine
>>>trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp16.engine --fp16 --explicitBatch --workspace=4G --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
# New Gammar for 8.5.0.12-1+cuda11.8
>>>trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp16.engine --fp16 --explicitBatch --memPoolSize=workspace:4096MiB --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
>>>trtexec --onnx=yolov8s-seg.onnx --saveEngine=yolov8s-seg_fp16.engine --fp16 --explicitBatch --memPoolSize=workspace:4096MiB --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw

# Install pycuda
>>>pip3 install pycuda==2023.1 tensorrt -i https://pypi.tuna.tsinghua.edu.cn/simple

Note: The yolov5-v7.0 models may experience inference errors in TensorRT-v7.x.x.x

"""

import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA 上下文

# 假设你有一个统一的基类 BackendBase
from .backend_base import BackendBase  # 请根据你的项目结构调整导入路径


class BackendTensorRT(BackendBase):
    """
    TensorRT 推理后端实现类。

    该类用于加载和执行已编译的 TensorRT 引擎文件（.engine），支持多输入/输出、动态形状，
    并提供与 ONNX Runtime 等后端统一的接口。使用字典格式传递输入输出数据，便于管理。
    """

    def __init__(self, model_path: str, **kwargs):
        """
        初始化 TensorRT 后端。

        Args:
            model_path (str): TensorRT 引擎文件路径（.engine）
            **kwargs: 其他可选参数（如日志级别等）
        """
        super().__init__(model_path, **kwargs)

        # TensorRT 相关对象
        self._engine: Optional[trt.ICudaEngine] = None  # 反序列化后的引擎
        self._context: Optional[trt.IExecutionContext] = None  # 执行上下文
        self._stream: Optional[cuda.Stream] = None  # CUDA 流，用于异步操作

        # 模型输入输出信息（以字典形式存储，便于查询）
        self.input_names: List[str] = []  # 输入张量名称列表
        self.output_names: List[str] = []  # 输出张量名称列表
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}  # 输入形状 {name: shape}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}  # 输出形状 {name: shape}
        self.input_dtypes: Dict[str, np.dtype] = {}  # 输入数据类型 {name: dtype}
        self.output_dtypes: Dict[str, np.dtype] = {}  # 输出数据类型 {name: dtype}

        # GPU 和 CPU 缓冲区（用于高效数据传输）
        self.host_inputs: Dict[str, np.ndarray] = {}  # CPU 页锁定内存（Host）
        self.device_inputs: Dict[str, cuda.DeviceAllocation] = {}  # GPU 内存（Device）
        self.host_outputs: Dict[str, np.ndarray] = {}
        self.device_outputs: Dict[str, cuda.DeviceAllocation] = {}

        # 绑定索引与名称的映射
        self.binding_name_to_index: Dict[str, int] = {}

    def _load_model(self, **kwargs) -> Any:
        """
        加载并初始化 TensorRT 引擎。

        步骤：
        1. 检查 .engine 文件是否存在
        2. 创建 TensorRT 运行时并反序列化引擎
        3. 创建执行上下文
        4. 提取输入输出信息
        5. 分配 GPU 和 CPU 缓冲区

        Returns:
            trt.ICudaEngine: 加载成功的引擎对象

        Raises:
            FileNotFoundError: 如果引擎文件不存在
            RuntimeError: 如果加载失败
        """
        # 1. 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorRT 引擎文件未找到: {self.model_path}")

        # 2. 创建 TensorRT logger（日志级别可根据 kwargs 调整）
        logger = trt.Logger(trt.Logger.WARNING)
        if kwargs.get("verbose", False):
            logger.min_severity = trt.Logger.SeverITY.VERBOSE

        # 3. 创建运行时并反序列化引擎
        with open(self.model_path, "rb") as f:
            runtime = trt.Runtime(logger)
            engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                raise RuntimeError("反序列化 TensorRT 引擎失败，数据可能损坏。")

        # 4. 创建执行上下文
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("创建 TensorRT 执行上下文失败。")

        # 5. 提取模型的输入输出信息
        self._extract_engine_info(engine)

        # 6. 分配输入输出缓冲区（Host + Device）
        self._allocate_buffers()

        # 7. 创建 CUDA 流（用于异步数据传输）
        self._stream = cuda.Stream()

        # 保存对象引用
        self._engine = engine
        self._context = context

        return engine  # 返回引擎对象

    def _extract_engine_info(self, engine: trt.ICudaEngine):
        """
        从 TensorRT 引擎中提取输入输出的名称、形状、数据类型等信息。

        Args:
            engine (trt.ICudaEngine): 已加载的 TensorRT 引擎
        """
        # 遍历所有绑定（binding），每个 binding 对应一个输入或输出张量
        for idx in range(engine.num_bindings):
            # 获取绑定名称
            name = engine.get_binding_name(idx)
            # 获取数据类型（转换为 NumPy 类型）
            dtype = trt.nptype(engine.get_binding_dtype(idx))
            # 获取形状
            shape = tuple(engine.get_binding_shape(idx))

            # 记录绑定名称与索引的映射
            self.binding_name_to_index[name] = idx

            if engine.binding_is_input(idx):
                # 是输入
                self.input_names.append(name)
                self.input_shapes[name] = shape
                self.input_dtypes[name] = dtype
            else:
                # 是输出
                self.output_names.append(name)
                self.output_shapes[name] = shape
                self.output_dtypes[name] = dtype

    def _allocate_buffers(self):
        """
        为所有输入和输出张量分配 Host（CPU）和 Device（GPU）内存缓冲区。

        使用页锁定内存（pinned memory）提高 Host ↔ Device 传输速度。
        """
        # 分配输入缓冲区
        for name in self.input_names:
            shape = self.input_shapes[name]
            dtype = self.input_dtypes[name]
            size = int(np.prod(shape))  # 总元素数量
            nbytes = size * np.dtype(dtype).itemsize  # 总字节数

            # 分配页锁定 Host 内存
            host_mem = cuda.pagelocked_empty(size, dtype=dtype)
            # 分配 Device 内存
            device_mem = cuda.mem_alloc(nbytes)

            self.host_inputs[name] = host_mem
            self.device_inputs[name] = device_mem

        # 分配输出缓冲区
        for name in self.output_names:
            shape = self.output_shapes[name]
            dtype = self.output_dtypes[name]
            size = int(np.prod(shape))
            nbytes = size * np.dtype(dtype).itemsize

            host_mem = cuda.pagelocked_empty(size, dtype=dtype)
            device_mem = cuda.mem_alloc(nbytes)

            self.host_outputs[name] = host_mem
            self.device_outputs[name] = device_mem

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行推理。

        Args:
            input_data (Dict[str, np.ndarray]): 输入数据字典，键为输入名称，值为 NumPy 数组

        Returns:
            Dict[str, np.ndarray]: 输出结果字典，键为输出名称，值为推理结果

        Raises:
            RuntimeError: 如果模型未加载或推理失败
            ValueError: 如果输入名称或形状不匹配
        """
        if not self._is_loaded:
            raise RuntimeError("模型未加载，请先调用 load() 方法。")

        # 1. 数据拷贝：将输入数据复制到 Host 缓冲区
        for name, data in input_data.items():
            if name not in self.input_names:
                raise ValueError(f"无效的输入名称: {name}，支持的输入: {self.input_names}")

            # 检查形状是否匹配（忽略 batch 维度）
            expected_shape = self.input_shapes[name]
            if data.shape != expected_shape:
                raise ValueError(f"输入 '{name}' 的形状不匹配。期望: {expected_shape}, 实际: {data.shape}")

            # 将数据展平并复制到 Host 缓冲区
            np.copyto(self.host_inputs[name], data.ravel())

        # 2. 异步传输：Host → Device（输入）
        for name in self.input_names:
            cuda.memcpy_htod_async(
                self.device_inputs[name],
                self.host_inputs[name],
                self._stream
            )

        # 3. 执行推理（异步）
        # 准备绑定地址列表（按 binding index 排序）
        bindings = []
        for name in self.input_names:
            bindings.append(int(self.device_inputs[name]))
        for name in self.output_names:
            bindings.append(int(self.device_outputs[name]))

        # 执行异步推理
        self._context.execute_async_v2(
            bindings=bindings,
            stream_handle=self._stream.handle
        )

        # 4. 异步传输：Device → Host（输出）
        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                self.host_outputs[name],
                self.device_outputs[name],
                self._stream
            )

        # 5. 同步流，等待所有操作完成
        self._stream.synchronize()

        # 6. 重塑输出数组为原始形状
        outputs = {}
        for name in self.output_names:
            shape = self.output_shapes[name]
            # 从展平数组恢复原始形状
            output_array = self.host_outputs[name].reshape(shape)
            outputs[name] = output_array

        return outputs

    def get_input_names(self) -> List[str]:
        """获取所有输入张量的名称。"""
        return self.input_names.copy()

    def get_output_names(self) -> List[str]:
        """获取所有输出张量的名称。"""
        return self.output_names.copy()

    def get_input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """获取输入张量的形状字典。"""
        return self.input_shapes.copy()

    def get_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """获取输出张量的形状字典。"""
        return self.output_shapes.copy()

    def get_input_dtypes(self) -> Dict[str, np.dtype]:
        """获取输入张量的数据类型字典。"""
        return self.input_dtypes.copy()

    def get_output_dtypes(self) -> Dict[str, np.dtype]:
        """获取输出张量的数据类型字典。"""
        return self.output_dtypes.copy()

    def get_metadata(self) -> Dict[str, str]:
        """
        获取模型元数据。

        注意：TensorRT 本身不支持自定义元数据。如果需要 author/version 等信息，
        建议搭配一个 .json 文件或在构建引擎时写入。
        """
        # 示例：可从外部文件读取，或硬编码
        return {
            "author": "unknown",
            "version": "1.0",
            "framework": "TensorRT"
        }

    def cleanup(self):
        """
        释放所有资源，防止 GPU 内存泄漏。
        """
        if self._context:
            self._context.__del__()
            self._context = None

        if self._engine:
            self._engine.__del__()
            self._engine = None

        if self._stream:
            self._stream = None  # pycuda 会自动清理

        # 清空缓冲区
        self.host_inputs.clear()
        self.device_inputs.clear()
        self.host_outputs.clear()
        self.device_outputs.clear()

        self._is_loaded = False
        print("TensorRT resources have been released.")

    def __enter__(self):
        """支持 with 语句进入。"""
        self.load()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """支持 with 语句退出，自动清理资源。"""
        self.cleanup()

    def __del__(self):
        """析构函数，作为资源清理的最后保障。"""
        if self._is_loaded:
            self.cleanup()
