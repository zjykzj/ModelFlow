# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/10 16:37
@File    : triton_verify.py
@Author  : zj
@Description: 
"""

# core/trt_verify.py

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


class TensorRTVerifier:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.stream = cuda.Stream()
        self.inputs, self.outputs, self.bindings = [], [], []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(binding)))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, input_data: np.ndarray):
        np.copyto(self.inputs[0]["host"], input_data.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        return [out["host"].reshape(self.engine.get_binding_shape(binding))
                for out, binding in zip(self.outputs, self.engine)]


if __name__ == "__main__":
    verifier = TensorRTVerifier("yolov5s.engine")
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    outputs = verifier.infer(dummy_input)
    print("TensorRT inference result shapes:", [o.shape for o in outputs])
