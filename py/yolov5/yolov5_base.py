# -*- coding: utf-8 -*-

"""
@date: 2024/1/2 下午9:11
@file: yolov5_base.py
@author: zj
@description: 
"""

import os
import cv2
import copy

import numpy as np
from numpy import ndarray

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from general import LOGGER, CLASSES_NAME
from numpy_util import draw_results, preprocess, postprocess


class YOLOv5Base:

    def __init__(self):
        super().__init__()

    def infer(self, im: ndarray):
        pass

    def detect(self, im0: ndarray):
        im = preprocess(im0)

        outputs = self.infer(im)

        boxes, confs, cls_ids = postprocess(outputs, im.shape[2:], im0.shape[:2], conf=0.25, iou=0.45)
        return boxes, confs, cls_ids

    def predict_image(self, img_path, output_dir="output/", save=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        im0 = cv2.imread(img_path)
        boxes, confs, cls_ids = self.detect(copy.deepcopy(im0))
        print(f"There are {len(boxes)} objects.")

        overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        if save:
            img_path = os.path.join(output_dir, f"{image_name}-yolov5_runtime_w_numpy.jpg")
            print(f"Save to {img_path}")
            cv2.imwrite(img_path, overlay)

    def predict_video(self, video_file, output_dir="output/", save=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        capture = cv2.VideoCapture(video_file)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"video_fps: {video_fps}, frame_count: {frame_count}, frame_width: {frame_width}, frame_height: {frame_height}")

        if save:
            image_name = os.path.splitext(os.path.basename(video_file))[0]
            video_out_name = f'{image_name}-yolov5_runtime_w_numpy.mp4'
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
            if save:
                writer.write(overlay)

            frame_id += 1
            print(f'frame_id: {frame_id}')

        if save:
            writer.release()
            print(f"Save to {video_path}")
