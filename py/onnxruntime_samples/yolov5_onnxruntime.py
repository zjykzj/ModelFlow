# -*- coding: utf-8 -*-

"""
@Time    : 2023/11/23 13:41
@File    : yolov5_onnxruntime.py
@Author  : zj
@Description: 
"""

import copy
import onnxruntime

import cv2
import numpy as np

from yolov5_util import letterbox, non_max_suppression, scale_boxes

print("ONNXRuntime version: {}".format(onnxruntime.__version__))


def preprocess(im0, fp16=False, img_size=640, stride=32, auto=False):
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

    if fp16:
        im = np.ascontiguousarray(im).astype(np.float16)
    else:
        im = np.ascontiguousarray(im).astype(np.float32)

    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    return im


class YOLOv5ONNXRuntime:

    def __init__(self, model, cuda=False):
        print(f'Loading {model} for ONNX Runtime inference...')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model, providers=providers)

        print("Input info:")
        x = self.session.get_inputs()[0]
        print("\t\t", x.name, x.shape)
        self.input_h = x.shape[2]
        self.input_w = x.shape[3]

        print("Output info:")
        self.output_names = []
        for x in self.session.get_outputs():
            print("\t\t", x.name, x.shape)
            self.output_names.append(x.name)

        self.dtype = self.session.get_inputs()[0].type
        print(f"Init Done. Work with {self.dtype}")

    def detect(self, im0,
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               half=False,  # use FP16 half-precision inference
               ):
        im = preprocess(im0, fp16=half, img_size=(self.input_w, self.input_h))

        preds = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})[0]
        # print(preds.shape)

        preds = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        return im, preds


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv5ONNXRuntime Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov5n.onnx',
                        help="ONNX Model Path")
    parser.add_argument("image", metavar="IMAGE", type=str, default="bus.jpg",
                        help="Image Path")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv5ONNXRuntime(args.model)
    im0 = cv2.imread(args.image)
    im, preds = model.detect(copy.deepcopy(im0))

    det = preds[0]
    print(f"det: {det.shape}")
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            x1 = int(xyxy[0].item())
            y1 = int(xyxy[1].item())
            x2 = int(xyxy[2].item())
            y2 = int(xyxy[3].item())
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    cv2.imwrite("yolov5_trt_out.jpg", im0)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
