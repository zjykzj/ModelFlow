# -*- coding: utf-8 -*-

"""
@date: 2024/1/7 下午9:43
@file: yolov8_base.py
@author: zj
@description:

Yolov8: https://github.com/ultralytics/ultralytics
Commit id: e58db228c2fd9856e7bff54a708bf5acde26fb29

"""

import os
import cv2
import copy

from tqdm import tqdm
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from general import CLASSES_NAME, LOGGER
from yolov8_util import LetterBox, draw_results
from yolov8_util import non_max_suppression, scale_boxes, check_imgsz


def pre_transform(im, imgsz, stride=32, pt=False):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = all(x.shape == im[0].shape for x in im)
    # print(f"imgsz: {imgsz}")
    # print(f"auto = {same_shapes and pt}")
    # print(f"stride = {stride}")
    letterbox = LetterBox(imgsz, auto=same_shapes and pt, stride=stride)
    return [letterbox(image=x) for x in im]


class YOLOv8Base(ABC):

    def __init__(self, imgsz=640, stride=32):
        super().__init__()
        imgsz = check_imgsz(imgsz, stride=stride, min_dim=2)
        self.imgsz = imgsz

    @abstractmethod
    def infer(self, im: ndarray):
        pass

    def preprocess(self, im, imgsz, stride=32, pt=False, fp16=False):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        # not_tensor = not isinstance(im, torch.Tensor)
        # if not_tensor:
        im = np.stack(pre_transform(im, imgsz, stride=stride, pt=pt))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        # im = torch.from_numpy(im)

        # im = im.to(device)
        # im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        # if not_tensor:
        im = im.astype(np.float16) if fp16 else im.astype(np.float32)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self,
                    preds,
                    img,
                    orig_imgs,
                    # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
                    conf=0.25,
                    iou=0.7,  # (float) intersection over union (IoU) threshold for NMS
                    agnostic_nms=False,  # (bool) class-agnostic NMS
                    max_det=300,  # (int) maximum number of detections per image
                    classes=None,
                    # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
                    ):
        # print("****************************************************")
        # print(f"conf= {conf}")
        # print(f"iou= {iou}")
        # print(f"agnostic_nms= {agnostic_nms}")
        # print(f"max_det= {max_det}")
        # print(f"classes= {classes}")
        # print("****************************************************")
        """Post-processes predictions and returns a list of Results objects."""
        preds = non_max_suppression(preds,
                                    conf_thres=conf,
                                    iou_thres=iou,
                                    agnostic=agnostic_nms,
                                    max_det=max_det,
                                    classes=classes)

        # if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        #     orig_imgs = convert_torch2numpy_batch(orig_imgs)

        # print(f"names= {self.model.names}")
        # results = []
        for i, pred in enumerate(preds):
            # print(f"img_path = {self.batch[0][i]}")
            orig_img = orig_imgs[i]
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # img_path = self.batch[0][i]
            # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))

        # print(f"result len: {len(results)}")
        # for result in results:
        #     print(result.boxes)
        # return results

        pred = preds[0]
        boxes = pred[:, :4]
        confs = pred[:, 4:5]
        cls_ids = pred[:, 5:6]
        return boxes, confs, cls_ids

    def detect(self, im0: ndarray):
        im0s = [im0]
        im = self.preprocess(im0s, self.imgsz)

        preds = self.infer(im)

        boxes, confs, cls_ids = self.postprocess(preds, im, im0s)
        # boxes, confs, cls_ids = postprocess(outputs, im.shape[2:], im0.shape[:2], conf=0.25, iou=0.45)
        return boxes, confs, cls_ids

    def predict_image(self, img_path, output_dir="output/", suffix="yolov8", save=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        im0 = cv2.imread(img_path)
        boxes, confs, cls_ids = self.detect(copy.deepcopy(im0))
        LOGGER.info(f"There are {len(boxes)} objects.")

        overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        if save:
            img_path = os.path.join(output_dir, f"{image_name}-{suffix}.jpg")
            LOGGER.info(f"Save to {img_path}")
            cv2.imwrite(img_path, overlay)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov8", save=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        capture = cv2.VideoCapture(video_file)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        LOGGER.info(
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
            LOGGER.info(f"Save to {video_path}")
