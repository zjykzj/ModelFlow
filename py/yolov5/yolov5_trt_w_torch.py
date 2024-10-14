# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/24 14:32
@File    : yolov8_trt_w_torch.py
@Author  : zj
@Description:

Usage - Infer Image/Video using YOLOv5 with TensorRT and Pytorch:
    $ python3 py/yolov5/yolov5_trt_w_torch.py yolov5s.engine assets/bus.jpg
    $ python3 py/yolov5/yolov5_trt_w_torch.py yolov5s.engine assets/vtest.avi --video

"""

import torch
from torch import Tensor

import numpy as np
from numpy import ndarray

from yolov5_base import YOLOv5Base

from general import LOGGER
from yolov5_util import letterbox
from torch_util import non_max_suppression, scale_boxes
from py.backends.backend_tensorrt import BackendTensorRT


class YOLOv5TRT(YOLOv5Base):

    def __init__(self, weight: str = 'yolov5s.engine'):
        super().__init__()
        self.session = BackendTensorRT(weight)
        self.device = torch.device("cuda")

    def infer(self, im: Tensor):
        im = im.cpu().numpy()  # torch to numpy

        y = self.session(im)

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

    def preprocess(self, im0, img_size=640, stride=32, auto=False, device=None, fp16=False):
        # return super().preprocess(im0, img_size, stride, auto)
        im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(device)
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im

    def postprocess(self, preds, im_shape, im0_shape, conf=0.25, iou=0.45, classes=None, agnostic=False, max_det=300):
        # return super().postprocess(preds, im_shape, im0_shape, conf, iou, classes, agnostic, max_det)
        # print("********* NMS START ***********")
        pred = non_max_suppression(preds, conf, iou, classes, agnostic, max_det=max_det)[0]
        # print("********* NMS END *************")

        boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
        confs = pred[:, 4:5]
        cls_ids = pred[:, 5:6]
        return boxes, confs, cls_ids

    def detect(self, im0: ndarray, conf=0.25, iou=0.45):
        return super().detect(im0, conf, iou)

    def predict_image(self, img_path, output_dir="output/", suffix="yolov5_trt_w_torch", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov5_trt_w_torch", save=False):
        super().predict_video(video_file, output_dir, suffix, save)


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv5TRT Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov5s.engine',
                        help="Path of TensorRT engine")
    parser.add_argument("input", metavar="INPUT", type=str, default="assets/bus.jpg",
                        help="Path of input, default to image")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Use video as input")

    parser.add_argument("--save", action="store_true", default=False,
                        help="Save or not.")

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")

    return args


def main(args):
    model = YOLOv5TRT(args.model)
    if args.video:
        model.predict_video(args.input, save=args.save)
    else:
        model.predict_image(args.input, save=args.save)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
