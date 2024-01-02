# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/24 14:32
@File    : yolov8_trt_w_torch.py
@Author  : zj
@Description:

Yolov5: https://github.com/ultralytics/yolov5
Commit id: 8d4058e093249cb538bc31a5248045cffb6d60af
Tag: v7.0

"""

import os
import cv2
import copy

import torch
from torch import Tensor

import numpy as np
from numpy import ndarray

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from general import LOGGER, letterbox
from numpy_util import draw_results
from torch_util import non_max_suppression, scale_boxes


def preprocess(im0, device, fp16=False, img_size=640, stride=32, auto=False):
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).to(device)
    im = im.half() if fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    return im


def postprocess(preds,
                im_shape,  # [h, w]
                im0_shape,  # [h, w]
                conf=0.25,
                iou=0.45,
                classes=None,
                agnostic=False,
                max_det=300, ):
    print("********* NMS START ***********")
    pred = non_max_suppression(preds, conf, iou, classes, agnostic, max_det=max_det)[0]
    print("********* NMS END *************")

    boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
    confs = pred[:, 4:5]
    cls_ids = pred[:, 5:6]
    return boxes, confs, cls_ids


class YOLOv5Runtime:

    def __init__(self, weight: str = 'yolov5s.onnx'):
        super().__init__()
        self.load_onnx(weight)

        self.device = torch.device('cpu')

    def load_onnx(self, weight: str):
        assert os.path.isfile(weight), weight

        LOGGER.info(f'Loading {weight} for ONNX Runtime inference...')
        import onnxruntime
        providers = ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(weight, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        metadata = session.get_modelmeta().custom_metadata_map  # metadata
        LOGGER.info(f"metadata: {metadata}")

        self.session = session
        self.output_names = output_names
        self.dtype = np.float32
        LOGGER.info(f"Init Done. Work with {self.dtype}")

    def infer(self, im: Tensor):
        im = im.cpu().numpy()  # torch to numpy

        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def detect(self, im0: ndarray):
        im = preprocess(im0, device=self.device)

        outputs = self.infer(im)
        np.save("outputs.npy", outputs)

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
        img_path = os.path.join(output_dir, f"{image_name}-yolov5_runtime_with_torch.jpg")
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
        video_out_name = f'{image_name}-yolov5_runtime_with_torch.mp4'
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

    parser = argparse.ArgumentParser(description="YOLOv5Runtime Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov5s.onnx',
                        help="Path of ONNX Runtime model")
    parser.add_argument("input", metavar="INPUT", type=str, default="assets/bus.jpg",
                        help="Path of input, default to image")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Use video as input")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv5Runtime(args.model)

    input = args.input
    if args.video:
        model.predict_video(input)
    else:
        model.predict_image(input)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
