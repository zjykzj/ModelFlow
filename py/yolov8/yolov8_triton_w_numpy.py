# -*- coding: utf-8 -*-

"""
@date: 2024/10/14 下午9:50
@file: yolov8_triton_w_numpy.py
@author: zj
@description:

Usage - Launch Triton Server by Docker in CPU mode:
    1. docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3
    2. /opt/tritonserver/bin/tritonserver --model-repository=./assets/model_repositories/triton_onnxruntime/ --model-control-mode=explicit

Usage - Infer Image/Video using YOLOv8 with Triton and Numpy:
    $ python3 py/yolov8/yolov8_triton_w_numpy.py assets/model_repositories/configs/DET_YOLOv8n.yaml assets/zidane.jpg
    $ python3 py/yolov8/yolov8_triton_w_numpy.py assets/model_repositories/configs/DET_YOLOv8n.yaml assets/vtest.avi --video

"""

import os
import yaml
from numpy import ndarray

from yolov8_base import YOLOv8Base

from general import LOGGER
from py.backends.backend_triton import BackendTriton
from py.backends.triton_client import TritonClientFactory


class YOLOv8Triton(YOLOv8Base):

    def __init__(self, model_name, input_name, output_name, is_fp16=False, grpc_client_url="localhost:8001",
                 imgsz=640, stride=32):
        super().__init__(imgsz, stride)
        triton_client = TritonClientFactory.get_client(url=grpc_client_url)
        assert triton_client is not None, f"triton_client should be provided, but got {triton_client}"

        self.session = BackendTriton(triton_client, model_name, input_name, output_name, is_fp16)

    def infer(self, im: ndarray):
        return self.session(im)

    def preprocess(self, im, imgsz, stride=32, pt=False, fp16=False):
        return super().preprocess(im, imgsz, stride, pt, fp16)

    def postprocess(self, preds, img, orig_imgs, conf=0.25, iou=0.7, agnostic_nms=False, max_det=300, classes=None):
        return super().postprocess(preds, img, orig_imgs, conf, iou, agnostic_nms, max_det, classes)

    def detect(self, im0: ndarray):
        return super().detect(im0)

    def predict_image(self, img_path, output_dir="output/", suffix="yolov8_triton_w_numpy", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov8_triton_w_numpy", save=False):
        super().predict_video(video_file, output_dir, suffix, save)


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8Triton Infer")
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

    model = YOLOv8Triton(model_name, input_name, output_name, is_fp16=is_fp16, grpc_client_url=args.url)
    if args.video:
        model.predict_video(args.input, save=args.save)
    else:
        model.predict_image(args.input, save=args.save)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
