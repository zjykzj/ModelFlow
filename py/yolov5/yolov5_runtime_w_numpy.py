# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/24 14:32
@File    : yolov8_trt_w_torch.py
@Author  : zj
@Description:
"""

import os

import numpy as np
from numpy import ndarray

from yolov5_base import YOLOv5Base
from general import LOGGER


class YOLOv5Runtime(YOLOv5Base):

    def __init__(self, weight: str = 'yolov5s.onnx'):
        super().__init__()
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
        im = im.astype(self.dtype)
        preds = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        return preds

    def detect(self, im0: ndarray, conf=0.25, iou=0.45):
        return super().detect(im0, conf, iou)

    def predict_image(self, img_path, output_dir="output/", suffix="yolov5_runtime_w_numpy", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov5_runtime_w_numpy", save=False):
        super().predict_video(video_file, output_dir, suffix, save)


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
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv5Runtime(args.model)

    input = args.input
    if args.video:
        model.predict_video(input, save=args.save)
    else:
        model.predict_image(input, save=args.save)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
