# -*- coding: utf-8 -*-

"""
@Time    : 2023/10/30 18:02
@File    : yolov8_trt.py
@Author  : zj
@Description: 
"""

import cv2
import numpy as np

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from yolov8_utils import det_process_box_output


def preprocess(image_raw, input_h, input_w, is_fp16=False):
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_w, input_h))
    image = image.transpose(2, 0, 1) / 255.0
    image = image[np.newaxis, :, :, :]
    if is_fp16:
        image = np.ascontiguousarray(image).astype(np.float16)
    else:
        image = np.ascontiguousarray(image).astype(np.float32)
    return image


def post_process(outputs, input_h, input_w, conf_thres=0.25, iou_thres=0.45, origin_w=0, origin_h=0):
    boxes, confs, classes = det_process_box_output(outputs[0],
                                                   conf_thres, iou_thres,
                                                   input_h, input_w,
                                                   origin_h, origin_w)
    return boxes, confs, classes


class YOLOv8TRT:

    def __init__(self, model):
        # Load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        with open(model, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
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
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.output_shapes.append(engine.get_binding_shape(binding))
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def inference(self, img):
        # Copy input image to host buffer
        self.inputs[0]['host'] = np.ravel(img)
        # Transfer input data  to the GPU.
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # Transfer input data  to the GPU.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

    def detect(self, img):
        h, w, c = img.shape
        reshaped = []

        processed_img = preprocess(img, self.input_h, self.input_w)
        outputs = self.inference(processed_img)
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))
        boxes, confs, classes = post_process(reshaped, self.input_h, self.input_w,
                                             conf_thres=0.4, iou_thres=0.4, origin_w=w, origin_h=h)
        return boxes, confs, classes


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8TRT Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov8n.engine',
                        help="TensorRT Engine Path")
    parser.add_argument("image", metavar="IMAGE", type=str, default="bus.jpg",
                        help="Image Path")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv8TRT(args.model)
    img = cv2.imread(args.image)
    boxes, confs, classes = model.detect(img)
    print(boxes, confs, classes)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
