# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/13 15:26
@File    : ultarlytics_export.py
@Author  : zj
@Description:

See:
1. https://docs.ultralytics.com/modes/export/
2. [Deploy YOLOv8 on NVIDIA Jetson using TensorRT](https://wiki.seeedstudio.com/YOLOv8-TRT-Jetson/)

Usage: Convert yolov8s to ONNX:
    $ python ultarlytics_export.py yolov8s
    $ python ultarlytics_export.py yolov8s-seg

Usage: Set opset=12:
    $ python ultarlytics_export.py yolov8s --opset 12
    $ python ultarlytics_export.py yolov8s-seg --opset 12

Note 1: After obtaining the onnx file, it can be converted into a trt file using the trtexec command-line tool
Note 2: For TensorRT 7.X version, setting opset=12 can perform the correct conversion and inference
Note 3: Ultralytics provides the command-line tool `yolo`

    $ yolo export model=yolov8s.pt format=onnx imgsz=640,640

"""

import torch
import argparse

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
    """Return the second-most recent ONNX opset version supported by current PyTorch."""
    opsets = [k for k in vars(torch.onnx) if k.startswith('symbolic_opset')]
    if not opsets:
        raise RuntimeError("Could not find supported ONNX opsets in torch.onnx.")
    max_opset = max(int(k[14:]) for k in opsets)
    return max_opset - 1  # Use second-most recent for stability


def parse_opt():
    parser = argparse.ArgumentParser(description="PyTorch to ONNX exporter for YOLOv8")
    parser.add_argument(
        "model",
        metavar="MODEL",
        type=str,
        default="yolov8n",
        choices=[
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg',
            'yolov8n-cls', 'yolov8s-cls', 'yolov8m-cls', 'yolov8l-cls', 'yolov8x-cls',
            'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose',
            'yolov8x-pose-p6'
        ],
        help="Model name (e.g. yolov8n, yolov8s-pose). Will download automatically if not found."
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version. Uses highest stable version if not specified."
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")
    return args


def main(args):
    model = YOLO(args.model)  # Auto-downloads if not cached
    model.export(format="onnx", opset=args.opset or get_latest_opset())


if __name__ == "__main__":
    main(parse_opt())
