# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/16 14:32
@File    : yolov8_trt_w_torch.py
@Author  : zj
@Description:

Yolov8: https://github.com/ultralytics/ultralytics
Commit id: e58db228c2fd9856e7bff54a708bf5acde26fb29

"""

import os
import cv2
import copy

import torch
from torch import Tensor

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
from yolov8_util import draw_results, LetterBox
from torch_util import check_imgsz, non_max_suppression, scale_boxes, convert_torch2numpy_batch
from yolov8_base import pre_transform


def preprocess(im, imgsz, device, stride=32, pt=False, fp16=False):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im, imgsz, stride=stride, pt=pt))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(device)
    im = im.half() if fp16 else im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def postprocess(preds,
                img,
                orig_imgs,
                # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
                conf=0.25,
                iou=0.7,  # (float) intersection over union (IoU) threshold for NMS
                agnostic_nms=False,  # (bool) class-agnostic NMS
                max_det=300,  # (int) maximum number of detections per image
                classes=None,  # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
                ):
    print("****************************************************")
    print(f"conf= {conf}")
    print(f"iou= {iou}")
    print(f"agnostic_nms= {agnostic_nms}")
    print(f"max_det= {max_det}")
    print(f"classes= {classes}")
    print("****************************************************")
    """Post-processes predictions and returns a list of Results objects."""
    preds = non_max_suppression(preds,
                                conf_thres=conf,
                                iou_thres=iou,
                                agnostic=agnostic_nms,
                                max_det=max_det,
                                classes=classes)

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = convert_torch2numpy_batch(orig_imgs)
    print('#' * 20)
    print(preds[0].reshape(-1)[:20])

    # print(f"names= {self.model.names}")
    # results = []
    for i, pred in enumerate(preds):
        # print(f"img_path = {self.batch[0][i]}")
        orig_img = orig_imgs[i]
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        # img_path = self.batch[0][i]
        # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))

    # print(f"result len: {len(results)}")
    # for result in results:
    #     print(result.boxes)
    # return results
    # return preds

    pred = preds[0]
    boxes = pred[:, :4]
    confs = pred[:, 4:5]
    cls_ids = pred[:, 5:6]
    return boxes, confs, cls_ids


class YOLOv8Runtime:

    def __init__(self, weight: str = 'yolov8n.onnx', imgsz=640, stride=32, device=torch.device("cpu")):
        super().__init__()
        self.load_onnx(weight)

        imgsz = check_imgsz(imgsz, stride=stride, min_dim=2)
        self.imgsz = imgsz

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

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

    def infer(self, im: Tensor):
        im = im.cpu().numpy()  # torch to numpy

        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})

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

    def detect(self, im0: ndarray):
        im0s = [im0]
        im = preprocess(im0s, self.imgsz, self.device)

        preds = self.infer(im)
        print("*" * 20)
        print(preds.reshape(-1)[:20])

        boxes, confs, cls_ids = postprocess(preds, im, im0s)
        # print(len(preds), preds[0].shape)
        return boxes, confs, cls_ids

        # boxes, confs, cls_ids = postprocess(outputs, im.shape[2:], im0.shape[:2], conf=0.25, iou=0.45)
        # return boxes, confs, cls_ids

    def predict_image(self, img_path, output_dir="output/"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        im0 = cv2.imread(img_path)
        boxes, confs, cls_ids = self.detect(copy.deepcopy(im0))
        print(f"There are {len(boxes)} objects.")

        overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        img_path = os.path.join(output_dir, f"{image_name}-yolov8_runtime_with_torch_out.jpg")
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
        video_out_name = f'{image_name}-yolov8_runtime_with_torch_out.mp4'
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

    parser = argparse.ArgumentParser(description="YOLOv8Runtime Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov8n.engine',
                        help="Path of ONNX Runtime model")
    parser.add_argument("input", metavar="INPUT", type=str, default="assets/bus.jpg",
                        help="Path of input, default to image")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Use video as input")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def main(args):
    model = YOLOv8Runtime(args.model)

    input = args.input
    if args.video:
        model.predict_video(input)
    else:
        model.predict_image(input)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
