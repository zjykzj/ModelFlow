# -*- coding: utf-8 -*-

"""
@Time    : 2024/9/7 16:19
@File    : backend_tensorrt_7x.py
@Author  : zj
@Description:

# Start Docker Container
>>>docker run -it --runtime nvidia --gpus=all --shm-size=16g -v /etc/localtime:/etc/localtime -v $(pwd):/workdir --workdir=/workdir --name tensorrt-v7.x nvcr.io/nvidia/pytorch:20.12-py3
>>>docker run -it --runtime nvidia --gpus=all --shm-size=16g -v /etc/localtime:/etc/localtime -v $(pwd):/workdir --workdir=/workdir --name tensorrt-v8.x ultralytics/yolov5:v7.0
# Convert onnx to engine
>>>trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp16.engine --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
>>>trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
# Install pycuda
>>>pip3 install pycuda==2023.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

Note: The yolov5-v7.0 models may experience inference errors in TensorRT-v7.x.x.x

"""

import os

import numpy as np
from numpy import ndarray

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


class BackendTensorRT:

    def __init__(self, weight: str = 'yolov5s.engine'):
        super().__init__()
        self.load_engine(weight)

    def load_engine(self, weight: str):
        assert os.path.isfile(weight), weight
        print(f'Loading {weight} for TensorRT inference...')

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(weight, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate memory
        self.inputs, self.outputs, self.bindings, self.output_shapes = [], [], [], []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            print(f"{binding} {engine.get_binding_shape(binding)}")
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.output_shapes.append(engine.get_binding_shape(binding))
                self.outputs.append({'host': host_mem, 'device': device_mem})

        self.dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding)))
        print(f"Init Done. Work with {self.dtype}")

    def __call__(self, im: ndarray):
        # Copy input image to host buffer
        self.inputs[0]['host'] = np.ravel(im.astype(self.dtype))
        # Transfer input data to the GPU.
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # Transfer input data to the GPU.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        outputs = [out['host'] for out in self.outputs]
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))

        return reshaped
