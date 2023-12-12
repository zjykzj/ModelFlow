# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/12 16:09
@File    : yolov8_trt_with_numpy.py
@Author  : zj
@Description: 
"""
from typing import List
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
from image_util import LetterBox, draw_results
from numpy_util import det_process_box_output, scale_boxes

MODEL_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
               22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
               29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
               35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
               40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
               48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
               55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
               62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
               69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
               76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

CLASSES_NAME = [item[1] for item in MODEL_NAMES.items()]


def pre_transform(im, imgsz=(640, 640), auto=False, stride=32):
    letterbox = LetterBox(imgsz, auto=auto, stride=stride)
    return [letterbox(image=x) for x in im]


def preprocess(im, imgsz=(640, 640)):
    assert isinstance(im, List) and isinstance(im[0], ndarray)
    im = np.stack(pre_transform(im, imgsz=imgsz))
    # BGR -> RGB and [N, H, W, C] -> [N, C, H, W]
    im = im[..., ::-1].transpose((0, 3, 1, 2))

    im = np.ascontiguousarray(im)  # contiguous
    im = im.astype(np.float32)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def postprocess(preds,
                im_shape,  # [h, w]
                im0_shape,  # [h, w]
                conf=0.25,
                iou=0.45,
                classes=None,
                agnostic=False,
                max_det=300, ):
    boxes, confs, classes = det_process_box_output(preds[0], conf, iou, im_shape[0], im0_shape[1])

    boxes = scale_boxes(im_shape, boxes, im0_shape)
    return boxes, confs, classes


class YOLOv8TRT:

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

    def detect(self, im0: ndarray):
        im = preprocess([im0])

        outputs = self.infer(im)

        boxes, confs, cls_ids = postprocess(outputs, im.shape[2:], im0.shape[:2], conf=0.25, iou=0.45)
        return boxes, confs, cls_ids

    def predict_image(self, img_path, output_dir="output/"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        im0 = cv2.imread(img_path)
        boxes, confs, cls_ids = self.detect(copy.deepcopy(im0))
        print(f"There are {len(boxes)} objects.")

        overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        img_path = os.path.join(output_dir, f"{image_name}-yolov8_trt_with_numpy.jpg")
        print(f"Save to {img_path}")
        cv2.imwrite(img_path, overlay)

    def predict_video(self, video_file, output_dir="output/"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        capture = cv2.VideoCapture(video_file)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"video_fps: {video_fps}, frame_count: {frame_count}, frame_width: {frame_width}, frame_height: {frame_height}")

        image_name = os.path.splitext(os.path.basename(video_file))[0]
        video_out_name = f'{image_name}-yolov8_trt_with_numpy.mp4'
        video_path = os.path.join(output_dir, video_out_name)
        video_format = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*video_format)
        writer = cv2.VideoWriter(video_path, fourcc, video_fps, (frame_width, frame_height))

        frame_id = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            boxes, confs, classes = self.detect(frame)
            print(f"There are {len(boxes)} objects.")
            overlay = draw_results(frame, boxes, confs, classes, CLASSES_NAME, is_xyxy=True)
            writer.write(overlay)

            frame_id += 1
            print(f'frame_id: {frame_id}')

        writer.release()
        print(f"Save to {video_path}")


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8TRT Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov8n.engine',
                        help="Path of TensorRT engine")
    parser.add_argument("input", metavar="INPUT", type=str, default="assets/bus.jpg",
                        help="Path of input, default to image")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Use video as input")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv8TRT(args.model)

    input = args.input
    if args.video:
        model.predict_video(input)
    else:
        model.predict_image(input)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
