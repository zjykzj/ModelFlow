# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/16 16:09
@File    : yolov8_trt_w_numpy.py
@Author  : zj
@Description:

Usage: Infer Image/Video using YOLOv8 with ONNXRuntime and Numpy:
    $ python3 py/yolov8/yolov8_runtime_w_numpy.py yolov8n.onnx assets/bus.jpg
    $ python3 py/yolov8/yolov8_runtime_w_numpy.py yolov8n.onnx assets/vtest.avi --video

Usage: Save Image/Video:
    $ python3 py/yolov8/yolov8_runtime_w_numpy.py yolov8n.onnx assets/bus.jpg --save
    $ python3 py/yolov8/yolov8_runtime_w_numpy.py yolov8n.onnx assets/vtest.avi --video --save

"""
import os

import numpy as np
from numpy import ndarray

from general import LOGGER
from yolov8_base import YOLOv8Base


class YOLOv8Runtime(YOLOv8Base):

    def __init__(self, weight: str = 'yolov8n.onnx', imgsz=640, stride=32):
        super().__init__(imgsz, stride)
        self.load_onnx(weight)

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

    def infer(self, im: ndarray):
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        return y[0]

    def preprocess(self, im, imgsz, stride=32, pt=False, fp16=False):
        return super().preprocess(im, imgsz, stride, pt, fp16)

    def postprocess(self, preds, img, orig_imgs, conf=0.25, iou=0.7, agnostic_nms=False, max_det=300, classes=None):
        return super().postprocess(preds, img, orig_imgs, conf, iou, agnostic_nms, max_det, classes)

    def detect(self, im0: ndarray):
        return super().detect(im0)

    def predict_image(self, img_path, output_dir="output/", suffix="yolov8_runtime_w_numpy", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov8_runtime_w_numpy", save=False):
        super().predict_video(video_file, output_dir, suffix, save)


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8Runtime Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov8n.engine',
                        help="Path of ONNX Runtime model")
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
    model = YOLOv8Runtime(args.model)

    if args.video:
        model.predict_video(args.input, save=args.save)
    else:
        model.predict_image(args.input, save=args.save)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
