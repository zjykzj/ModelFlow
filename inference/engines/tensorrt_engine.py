# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:05
@File    : tensorrt_engine.py
@Author  : zj
@Description:

pip install nvidia-tensorrt pycuda

"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os


class TensorRTEngine:
    def __init__(self, model_path: str):
        """
        初始化 TensorRT 引擎
        :param model_path: .engine 文件路径
        """
        self.model_path = model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()

        # 分配输入输出内存
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _load_engine(self):
        """从文件加载 TensorRT 引擎"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件 {self.model_path} 不存在")

        with open(self.model_path, "rb") as f:
            engine_data = f.read()
        return self.runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self):
        """分配输入输出缓存"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append({'name': binding, 'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'name': binding, 'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        执行推理
        :param input_tensor: 输入 numpy 数组 (NCHW)
        :return: 输出 numpy 数组
        """
        # 将输入数据拷贝到输入缓存
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())

        # 将输入 host 数据复制到 device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 将输出从 device 拷贝回 host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)

        # 同步流
        self.stream.synchronize()

        # 返回输出结果
        return self.outputs[0]['host'].reshape(self.engine.get_binding_shape(self.outputs[0]['name']))
