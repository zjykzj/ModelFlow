# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 16:46
@File    : trt_verify.py
@Author  : zj
@Description: 
"""

import numpy as np
from typing import List

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


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


def load_dummy_input_from_onnx(onnx_path: str, batch_size: int = 1, dtype=np.float32) -> np.ndarray:
    import onnx
    model = onnx.load(onnx_path)
    input_shape = [dim.dim_value if dim.HasField('dim_value') else -1 for dim in
                   model.graph.input[0].type.tensor_type.shape.dim]
    assert input_shape[0] == -1 or input_shape[0] == batch_size, "Input shape mismatch with batch size."

    input_shape[0] = batch_size
    return np.random.rand(*input_shape).astype(dtype)


def verify_trt_output(onnx_path: str, engine_path: str, batch_size: int = 1, is_fp16: bool = False):
    """
    Verify that the TensorRT engine can run and produce output of correct shape.

    Does NOT use PyTorch or ONNXRuntime for comparison.
    """
    target_dtype = np.float16 if is_fp16 else np.float32
    dummy_input = load_dummy_input_from_onnx(onnx_path, batch_size=batch_size, dtype=target_dtype)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        np.copyto(inputs[0].host, dummy_input.ravel())
        trt_outputs = do_inference(context, bindings, inputs, outputs, stream, batch_size=batch_size)

    print("✅ TensorRT inference completed.")
    print("Output shape:", trt_outputs[0].shape)
    print("First 5 values:", trt_outputs[0][:5])
