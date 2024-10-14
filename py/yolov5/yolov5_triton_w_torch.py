# -*- coding: utf-8 -*-

"""
@date: 2024/10/14 下午9:41
@file: yolov5_triton_w_torch.py
@author: zj
@description:

Usage - Launch Triton Server by Docker in CPU mode:
    1. docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3
    2. /opt/tritonserver/bin/tritonserver --model-repository=./assets/model_repositories/triton_onnxruntime/ --model-control-mode=explicit

Usage - Infer Image/Video using YOLOv5 with Triton and Numpy:
    $ python3 py/yolov5/yolov5_triton_w_torch.py assets/model_repositories/configs/DET_YOLOv5n.yaml assets/bus.jpg
    $ python3 py/yolov5/yolov5_triton_w_torch.py assets/model_repositories/configs/DET_YOLOv5n.yaml assets/vtest.avi --video

"""

import os
import yaml

import torch
from torch import Tensor

import numpy as np
from numpy import ndarray

from yolov5_base import YOLOv5Base

from general import LOGGER
from yolov5_util import letterbox
from torch_util import non_max_suppression, scale_boxes
from py.backends.backend_triton import BackendTriton
from py.backends.triton_client import TritonClientFactory


class YOLOv5Triton(YOLOv5Base):

    def __init__(self, model_name, input_name, output_name, is_fp16=False, grpc_client_url="localhost:8001"):
        super().__init__()
        triton_client = TritonClientFactory.get_client(url=grpc_client_url)
        assert triton_client is not None, f"triton_client should be provided, but got {triton_client}"

        self.session = BackendTriton(triton_client, model_name, input_name, output_name, is_fp16)
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

    def predict_image(self, img_path, output_dir="output/", suffix="yolov5_triton_w_torch", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov5_triton_w_torch", save=False):
        super().predict_video(video_file, output_dir, suffix, save)


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv5Triton Infer")
    parser.add_argument("config", metavar='CONFIG', type=str, help="Path to config file")
    parser.add_argument("input", metavar="INPUT", type=str, help="Path of input, default to image")

    parser.add_argument("--url", metavar="URL", type=str, default="localhost:8001", help="URL")
    parser.add_argument("--video", action="store_true", default=False, help="Use video as input")
    parser.add_argument("--save", action="store_true", default=False, help="Save or not.")

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")

    return args


def main(args):
    config_file = args.config
    assert os.path.isfile(config_file), config_file
    with open(config_file, "r") as f:
        cfg_dict = yaml.safe_load(f)

    model_name = cfg_dict['MODEL_NAME']
    input_name = cfg_dict['INPUT_NAME']
    output_name = cfg_dict['OUTPUT_NAME']
    is_fp16 = cfg_dict['FP16']

    model = YOLOv5Triton(model_name, input_name, output_name, is_fp16=is_fp16, grpc_client_url=args.url)
    if args.video:
        model.predict_video(args.input, save=args.save)
    else:
        model.predict_image(args.input, save=args.save)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
