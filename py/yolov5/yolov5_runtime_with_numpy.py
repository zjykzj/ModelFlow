# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/12 14:32
@File    : yolov8_trt_with_torch.py
@Author  : zj
@Description:

Yolov5: https://github.com/ultralytics/yolov5
Commit id: 915bbf294bb74c859f0b41f1c23bc395014ea679
Tag: v7.0

"""

import os
import cv2
import copy

import numpy as np
from numpy import ndarray

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from general import LOGGER, CLASSES_NAME
from numpy_util import draw_results, preprocess, postprocess


class YOLOv5Runtime:

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

    def detect(self, im0: ndarray):
        im = preprocess(im0)

        outputs = self.infer(im)

        boxes, confs, cls_ids = postprocess(outputs, im.shape[2:], im0.shape[:2], conf=0.25, iou=0.45)
        return boxes, confs, cls_ids

    def predict_image(self, img_path, output_dir="output/"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        im0 = cv2.imread(img_path)
        boxes, confs, cls_ids = self.detect(copy.deepcopy(im0))
        print(f"There are {len(boxes)} objects.")

        overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        img_path = os.path.join(output_dir, f"{image_name}-yolov5_runtime_with_numpy.jpg")
        print(f"Save to {img_path}")
        cv2.imwrite(img_path, overlay)

    def predict_video(self, video_file, output_dir="output/"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        capture = cv2.VideoCapture(video_file)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"video_fps: {video_fps}, frame_count: {frame_count}, frame_width: {frame_width}, frame_height: {frame_height}")

        image_name = os.path.splitext(os.path.basename(video_file))[0]
        video_out_name = f'{image_name}-yolov5_runtime_with_numpy.mp4'
        video_path = os.path.join(output_dir, video_out_name)
        video_format = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*video_format)
        writer = cv2.VideoWriter(video_path, fourcc, video_fps, (frame_width, frame_height))

        frame_id = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            boxes, confs, classes = self.detect(frame)
            print(f"There are {len(boxes)} objects.")
            overlay = draw_results(frame, boxes, confs, classes, CLASSES_NAME, is_xyxy=True)
            writer.write(overlay)

            frame_id += 1
            print(f'frame_id: {frame_id}')

        writer.release()
        print(f"Save to {video_path}")


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv5Runtime Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov8n.onnx',
                        help="Path of ONNX Runtime model")
    parser.add_argument("input", metavar="INPUT", type=str, default="assets/bus.jpg",
                        help="Path of input, default to image")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Use video as input")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv5Runtime(args.model)

    input = args.input
    if args.video:
        model.predict_video(input)
    else:
        model.predict_image(input)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
