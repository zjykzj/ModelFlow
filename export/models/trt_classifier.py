# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 19:46
@File    : trt_classifier.py
@Author  : zj
@Description: 
"""

import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}"

    def __repr__(self):
        return self.__str__()


class TRTClassifier:
    def __init__(self, engine_path: str):
        """
        Load a TensorRT engine and prepare for inference.
        Also prints detailed model info including input/output shapes, data types, etc.

        Args:
            engine_path (str): Path to the .engine file.
        """
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        print(f"✅ TensorRT engine loaded from {engine_path}")
        self._print_engine_info()

    def _print_engine_info(self):
        """Print detailed information about the TensorRT engine."""
        print("🔍 TensorRT Engine Info:")
        print(f"  Engine Name:      {self.engine.name}")
        print(f"  Max Batch Size:   {self.engine.max_batch_size}")
        print(f"  Device Memory Size: {self.engine.device_memory_size / (1024 ** 2):.2f} MB")

        print(f"  Inputs:")
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                name = self.engine.get_binding_name(idx)
                dtype = self.engine.get_binding_dtype(name)
                shape = list(self.engine.get_binding_shape(name))
                dtype_str = trt.nptype(dtype).__name__  # ✅ Fix here
                print(f"    {name} | Shape: {shape} | Data Type: {dtype_str}")

        print(f"  Outputs:")
        for idx in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(idx):
                name = self.engine.get_binding_name(idx)
                dtype = self.engine.get_binding_dtype(name)
                shape = list(self.engine.get_binding_shape(name))
                dtype_str = trt.nptype(dtype).__name__  # ✅ Fix here
                print(f"    {name} | Shape: {shape} | Data Type: {dtype_str}")

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the input data.

        Args:
            input_data (np.ndarray): Input tensor as a NumPy array.

        Returns:
            np.ndarray: Model output.
        """
        # Copy input data to GPU
        np.copyto(self.inputs[0].host, input_data.ravel())
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output back
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()

        return self.outputs[0].host.copy()
