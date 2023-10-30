# -*- coding: utf-8 -*-

"""
@Time    : 2023/10/30 14:47
@File    : yolov8_export.py
@Author  : zj
@Description:
See:
1. https://docs.ultralytics.com/modes/export/
2. [Deploy YOLOv8 on NVIDIA Jetson using TensorRT](https://wiki.seeedstudio.com/YOLOv8-TRT-Jetson/)

Usage: Convert yolov8n to ONNX:
    $ python yolov8_export.py yolov8n

Usage: Set opset=12:
    $ python yolov8_export.py yolov8n --opset=12

After obtaining the onnx file, it can be converted into a trt file using the trtexec command-line tool

# 单精度转换
>>>trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.trt --explicitBatch
# 半精度转换
>>>trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16

Note: For TensorRT 7.X version, setting opset=12 can perform the correct conversion

"""

import argparse

import torch

from ultralytics import YOLO

model_download_link = {
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',

    'yolov8n-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt',
    'yolov8s-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt',
    'yolov8m-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt',
    'yolov8l-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt',
    'yolov8x-seg': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt',

    'yolov8n-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt',
    'yolov8s-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt',
    'yolov8m-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt',
    'yolov8l-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt',
    'yolov8x-cls': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt',

    'yolov8n-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt',
    'yolov8s-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt',
    'yolov8m-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt',
    'yolov8l-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt',
    'yolov8x-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt',
    'yolov8x-pose-p6': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt'
}


def get_latest_opset():
    """Return second-most (for maturity) recently supported ONNX opset by this version of torch."""
    return max(int(k[14:]) for k in vars(torch.onnx) if 'symbolic_opset' in k) - 1  # opset


def parse_opt():
    parser = argparse.ArgumentParser(description="ONNX to Engine")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov8n', choices=model_download_link.keys(),
                        help="model name. Refer to https://docs.ultralytics.com/models/yolov8/#supported-modes")
    parser.add_argument("--opset", metavar="OPTSET", type=int, default=get_latest_opset(),
                        help="Onnx optimization settings, default to highest level")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    # Load a model
    model = YOLO(args.model)  # load a custom trained
    # print(model)
    model.export(format='onnx', opset=args.opset)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
