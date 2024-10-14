# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/24 14:32
@File    : yolov8_trt_w_torch.py
@Author  : zj
@Description:

Usage - Infer Image/Video using YOLOv5 with ONNXRuntime and Numpy:
    $ python3 py/yolov5/yolov5_runtime_w_numpy.py yolov5s.onnx assets/bus.jpg
    $ python3 py/yolov5/yolov5_runtime_w_numpy.py yolov5s.onnx assets/vtest.avi  --video

"""

from numpy import ndarray

from py.backends.backend_runtime import BackendRuntime

from yolov5_base import YOLOv5Base
from general import LOGGER


class YOLOv5Runtime(YOLOv5Base):

    def __init__(self, weight: str = 'yolov5s.onnx'):
        super().__init__()

        self.session = BackendRuntime(weight)

    def infer(self, im: ndarray):
        return self.session(im)

    def preprocess(self, im0, img_size=640, stride=32, auto=False):
        return super().preprocess(im0, img_size, stride, auto)

    def postprocess(self, preds, im_shape, im0_shape, conf=0.25, iou=0.45, classes=None, agnostic=False, max_det=300):
        return super().postprocess(preds, im_shape, im0_shape, conf, iou, classes, agnostic, max_det)

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
    LOGGER.info(f"args: {args}")

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
