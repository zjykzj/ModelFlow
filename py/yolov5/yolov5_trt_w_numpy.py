# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/24 14:32
@File    : yolov8_trt_w_torch.py
@Author  : zj
@Description:

Usage: Infer Image/Video using YOLOv5 with TensorRT and Numpy:
    $ python3 py/yolov5/yolov5_trt_w_numpy.py yolov5s.engine assets/bus.jpg
    $ python3 py/yolov5/yolov5_trt_w_numpy.py yolov5s.engine assets/vtest.avi --video

Usage: Save Image/Video:
    $ python3 py/yolov5/yolov5_trt_w_numpy.py yolov5s.engine assets/bus.jpg --save
    $ python3 py/yolov5/yolov5_trt_w_numpy.py yolov5s.engine assets/vtest.avi --video --save

"""

from numpy import ndarray

from yolov5_base import YOLOv5Base

from general import LOGGER


class YOLOv5TRT(YOLOv5Base):

    def __init__(self, weight: str = 'yolov5s.engine'):
        super().__init__()

        self.session = BackendTensorRT(weight)

    def infer(self, im: ndarray):
        return self.session(im)

    def preprocess(self, im0, img_size=640, stride=32, auto=False):
        return super().preprocess(im0, img_size, stride, auto)

    def postprocess(self, preds, im_shape, im0_shape, conf=0.25, iou=0.45, classes=None, agnostic=False, max_det=300):
        return super().postprocess(preds, im_shape, im0_shape, conf, iou, classes, agnostic, max_det)

    def detect(self, im0: ndarray, conf=0.25, iou=0.45):
        return super().detect(im0, conf, iou)

    def predict_image(self, img_path, output_dir="output/", suffix="yolov5_trt_w_numpy", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov5_trt_w_numpy", save=False):
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
    parser.add_argument("--v7", action='store_true', default=False,
                        help="Use TensorRT_v8.x.x.x or TensorRT_v7.x.x.x. Defaults to TensorRT_v8.x.x.x")

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
    if args.v7:
        print(f"Use TensorRT 7.x.x.x")
        from py.backends.backend_tensorrt_7x import BackendTensorRT_7x as BackendTensorRT
    else:
        print(f"Use TensorRT 8.x.x.x")
        from py.backends.backend_tensorrt_8x import BackendTensorRT_8x as BackendTensorRT

    main(args)
