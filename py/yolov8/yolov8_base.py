# -*- coding: utf-8 -*-

"""
@date: 2024/1/7 下午9:43
@file: yolov8_base.py
@author: zj
@description:

Yolov8: https://github.com/ultralytics/ultralytics
Commit id: ed73c0fedc53251eefaf19cdaec17827ea2cfbfb

"""

import os
import cv2
import copy

from tqdm import tqdm

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
from yolov8_util import draw_results, letterbox, non_max_suppression, scale_boxes


class YOLOv8Base:

    def __init__(self):
        super().__init__()

    def infer(self, im: ndarray):
        pass

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
        print(f"There are {len(boxes)} objects.")

        overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        if save:
            img_path = os.path.join(output_dir, f"{image_name}-{suffix}.jpg")
            print(f"Save to {img_path}")
            cv2.imwrite(img_path, overlay)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov5", save=False):
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
            video_out_name = f'{image_name}-{suffix}.mp4'
            video_path = os.path.join(output_dir, video_out_name)
            video_format = 'mp4v'
            fourcc = cv2.VideoWriter_fourcc(*video_format)
            writer = cv2.VideoWriter(video_path, fourcc, video_fps, (frame_width, frame_height))

        # frame_id = 0
        # while True:
        for _ in tqdm(range(frame_count)):
            ret, frame = capture.read()
            if not ret:
                break

            boxes, confs, classes = self.detect(frame)
            # print(f"There are {len(boxes)} objects.")
            overlay = draw_results(frame, boxes, confs, classes, CLASSES_NAME, is_xyxy=True)
            if save:
                writer.write(overlay)

            # frame_id += 1
            # print(f'frame_id: {frame_id}')

        if save:
            writer.release()
            print(f"Save to {video_path}")

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
                    agnostic_nms=False,
                    max_det=300, ):
        # print("********* NMS START ***********")
        pred = non_max_suppression(preds,
                                   conf,
                                   iou,
                                   agnostic=agnostic_nms,
                                   max_det=max_det,
                                   classes=classes)
        # print("********* NMS END *************")
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

        boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
        confs = pred[:, 4:5]
        cls_ids = pred[:, 5:6]
        return boxes, confs, cls_ids
