# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/24 14:32
@File    : yolov8_trt_w_torch.py
@Author  : zj
@Description:

docker run --gpus all -it --rm -v ${PWD}:/workdir --workdir=/workdir nvcr.io/nvidia/pytorch:20.12-py3 bash

Usage: Infer Image/Video using YOLOv5 with TensorRT and Numpy:
    $ python3 py/yolov5/yolov5_trt_w_numpy.py yolov5s.onnx assets/bus.jpg
    $ python3 py/yolov5/yolov5_trt_w_numpy.py yolov5s.onnx assets/bus.jpg  --video

Usage: Save Image/Video:
    $ python3 py/yolov5/yolov5_trt_w_numpy.py yolov5s.onnx assets/bus.jpg --save
    $ python3 py/yolov5/yolov5_trt_w_numpy.py yolov5s.onnx assets/vtest.avi --video --save

"""

import os
import cv2
import copy

import numpy as np
from numpy import ndarray

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from general import LOGGER
from yolov5_base import YOLOv5Base


class YOLOv8TRT(YOLOv5Base):

    def __init__(self, weight: str = 'yolov8n.engine'):
        super().__init__()
        self.load_engine(weight)

    def load_engine(self, weight: str):
        assert os.path.isfile(weight), weight
        LOGGER.info(f'Loading {weight} for TensorRT inference...')

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
            print(binding, engine.get_binding_shape(binding))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.output_shapes.append(engine.get_binding_shape(binding))
                self.outputs.append({'host': host_mem, 'device': device_mem})

        self.dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding)))
        LOGGER.info(f"Init Done. Work with {self.dtype}")

    def infer(self, im: ndarray):
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

    def detect(self, im0: ndarray, conf=0.25, iou=0.45):
        return super().detect(im0, conf, iou)

    def predict_image(self, img_path, output_dir="output/", suffix="yolov5_trt_w_numpy", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov5_trt_w_numpy", save=False):
        super().predict_video(video_file, output_dir, suffix, save)

    def preprocess(self, im0, img_size=640, stride=32, auto=False):
        return super().preprocess(im0, img_size, stride, auto)

    def postprocess(self, preds, im_shape, im0_shape, conf=0.25, iou=0.45, classes=None, agnostic=False, max_det=300):
        return super().postprocess(preds, im_shape, im0_shape, conf, iou, classes, agnostic, max_det)


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8TRT Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov8n.engine',
                        help="Path of TensorRT engine")
    parser.add_argument("input", metavar="INPUT", type=str, default="assets/bus.jpg",
                        help="Path of input, default to image")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Use video as input")

    parser.add_argument("--save", action="store_true", default=False,
                        help="Save or not.")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv8TRT(args.model)

    input = args.input
    if args.video:
        model.predict_video(input, save=args.save)
    else:
        model.predict_image(input, save=args.save)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
