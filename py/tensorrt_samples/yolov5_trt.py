# -*- coding: utf-8 -*-

"""
@Time    : 2023/11/23 11:11
@File    : yolov5_trt.py
@Author  : zj
@Description: YOLOv5_v7.0 + TensorRT_v8.2.4 + Numpy (without pytorch)

Usage: Specify YOLOv5 Engine model for TensorRT inference:
    $ python3 yolov5_trt.py yolov5n.engine bus.jpg

"""
import copy

import cv2
import numpy as np

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from yolov5_util import letterbox, non_max_suppression, scale_boxes

print("TensorRT version: {}".format(trt.__version__))


def preprocess(im0, fp16=False, img_size=640, stride=32, auto=False):
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

    if fp16:
        im = np.ascontiguousarray(im).astype(np.float16)
    else:
        im = np.ascontiguousarray(im).astype(np.float32)

    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    return im


class YOLOv5TRT:

    def __init__(self, model):
        # Load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        with open(model, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate memory
        self.inputs, self.outputs, self.bindings, self.output_shapes = [], [], [], []
        print_input, print_output = True, True
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # print(engine.get_binding_dtype(binding), dtyp e, np.dtype(dtype))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            binding_shape = engine.get_binding_shape(binding)
            if engine.binding_is_input(binding):
                self.input_w = binding_shape[-1]
                self.input_h = binding_shape[-2]
                self.inputs.append({'host': host_mem, 'device': device_mem})
                if print_input:
                    print("Input info:")
                    print_input = False
                print("\t\t", binding, binding_shape)
            else:
                self.output_shapes.append(binding_shape)
                self.outputs.append({'host': host_mem, 'device': device_mem})
                if print_output:
                    print("Output info:")
                    print_output = False
                print("\t\t", binding, binding_shape)

        self.dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding)))
        print(f"Init Done. Work with {self.dtype}")

    def inference(self, img):
        # Copy input image to host buffer
        self.inputs[0]['host'] = np.ravel(img.astype(self.dtype))
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

    def detect(self, im0,
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               half=False,  # use FP16 half-precision inference
               ):
        im = preprocess(im0, fp16=half, img_size=(self.input_w, self.input_h))

        outputs = self.inference(im)[-1]

        reshaped = outputs.reshape(self.output_shapes[-1])
        preds = non_max_suppression(reshaped, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        return im, preds


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv5TRT Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov5n.engine',
                        help="TensorRT Engine Path")
    parser.add_argument("image", metavar="IMAGE", type=str, default="bus.jpg",
                        help="Image Path")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv5TRT(args.model)
    im0 = cv2.imread(args.image)
    im, preds = model.detect(copy.deepcopy(im0))

    det = preds[0]
    print(f"det: {det.shape}")
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            x1 = int(xyxy[0].item())
            y1 = int(xyxy[1].item())
            x2 = int(xyxy[2].item())
            y2 = int(xyxy[3].item())
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    cv2.imwrite("yolov5_trt_out.jpg", im0)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
