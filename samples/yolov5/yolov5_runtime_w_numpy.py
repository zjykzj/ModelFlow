# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/29 16:47
@File    : yolov5_runtime_w_numpy.py
@Author  : zj
@Description:
"""

import os
import cv2
import copy

from tqdm import tqdm

import numpy as np
from numpy import ndarray
import logging

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# 获取当前工作目录
CURRENT_DIR = Path.cwd()
# 将当前工作目录添加到sys.path
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from core.backends.backend_runtime import BackendRuntime
from general import CLASSES_NAME, draw_results
from numpy_util import letterbox, non_max_suppression, scale_boxes


class YOLOv5Runtime:

    def __init__(self, weight: str = 'yolov5s.onnx'):
        super().__init__()

        self.session = BackendRuntime(weight)

    def infer(self, im: ndarray):
        return self.session(im)

    def preprocess(self, im0, img_size=640, stride=32, auto=False):
        im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = im.astype(float)  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im

    def postprocess(self,
                    preds,
                    im_shape,  # [h, w]
                    im0_shape,  # [h, w]
                    conf=0.25,
                    iou=0.45,
                    classes=None,
                    agnostic=False,
                    max_det=300, ):
        # print("********* NMS START ***********")
        pred = non_max_suppression(preds[0], conf, iou, classes, agnostic, max_det=max_det)[0]
        # print("********* NMS END *************")

        boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
        confs = pred[:, 4:5]
        cls_ids = pred[:, 5:6]
        return boxes, confs, cls_ids

    def detect(self, im0: ndarray, conf=0.25, iou=0.45):
        im = self.preprocess(im0)

        outputs = self.infer(im)

        boxes, confs, cls_ids = self.postprocess(outputs, im.shape[2:], im0.shape[:2], conf=conf, iou=iou)
        return boxes, confs, cls_ids

    def predict_image(self, img_path, output_dir="output/", suffix="yolov5", save=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        im0 = cv2.imread(img_path)
        boxes, confs, cls_ids = self.detect(copy.deepcopy(im0))
        logging.info(f"There are {len(boxes)} objects.")

        overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        if save:
            img_path = os.path.join(output_dir, f"{image_name}-{suffix}.jpg")
            logging.info(f"Save to {img_path}")
            cv2.imwrite(img_path, overlay)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov5", save=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        capture = cv2.VideoCapture(video_file)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(
            f"video_fps: {video_fps}, frame_count: {frame_count}, frame_width: {frame_width}, frame_height: {frame_height}")

        if save:
            image_name = os.path.splitext(os.path.basename(video_file))[0]
            video_out_name = f'{image_name}-{suffix}.mp4'
            video_path = os.path.join(output_dir, video_out_name)
            video_format = 'mp4v'
            fourcc = cv2.VideoWriter_fourcc(*video_format)
            writer = cv2.VideoWriter(video_path, fourcc, video_fps, (frame_width, frame_height))

        for _ in tqdm(range(frame_count)):
            ret, frame = capture.read()
            if not ret:
                break

            boxes, confs, classes = self.detect(frame)
            overlay = draw_results(frame, boxes, confs, classes, CLASSES_NAME, is_xyxy=True)
            if save:
                writer.write(overlay)

        if save:
            writer.release()
            logging.info(f"Save to {video_path}")


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv5Runtime Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov5s.onnx',
                        help="Path of ONNX Runtime model")
    parser.add_argument("input", metavar="INPUT", type=str, default="assets/bus.jpg",
                        help="Path of input, default to image")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Use video as input")

    parser.add_argument("--save", action="store_true", default=False,
                        help="Save or not.")

    args = parser.parse_args()
    logging.info(f"args: {args}")

    return args


def main(args):
    model = YOLOv5Runtime(args.model)

    if args.video:
        model.predict_video(args.input, save=args.save)
    else:
        model.predict_image(args.input, save=args.save)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
