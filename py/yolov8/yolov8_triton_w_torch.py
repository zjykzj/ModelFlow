# -*- coding: utf-8 -*-

"""
@date: 2024/10/14 下午9:59
@file: yolov8_triton_w_torch.py
@author: zj
@description:

Usage - Launch Triton Server by Docker in CPU mode:
    1. docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3
    2. /opt/tritonserver/bin/tritonserver --model-repository=./assets/model_repositories/triton_onnxruntime/ --model-control-mode=explicit

Usage - Infer Image/Video using YOLOv8 with Triton and Pytorch:
    $ python3 py/yolov8/yolov8_triton_w_torch.py assets/model_repositories/configs/DET_YOLOv8n.yaml assets/zidane.jpg
    $ python3 py/yolov8/yolov8_triton_w_torch.py assets/model_repositories/configs/DET_YOLOv8n.yaml assets/vtest.avi --video

"""
import os
import yaml
import torch
from torch import Tensor

import numpy as np
from numpy import ndarray

from yolov8_base import YOLOv8Base, pre_transform

from general import LOGGER
from torch_util import non_max_suppression, scale_boxes, convert_torch2numpy_batch
from py.backends.backend_triton import BackendTriton
from py.backends.triton_client import TritonClientFactory


class YOLOv8Triton(YOLOv8Base):

    def __init__(self, model_name, input_name, output_name, is_fp16=False, grpc_client_url="localhost:8001",
                 imgsz=640, stride=32, device=torch.device("cpu")):
        super().__init__(imgsz, stride)
        triton_client = TritonClientFactory.get_client(url=grpc_client_url)
        assert triton_client is not None, f"triton_client should be provided, but got {triton_client}"

        self.session = BackendTriton(triton_client, model_name, input_name, output_name, is_fp16)
        self.device = device

    def infer(self, im: Tensor):
        im = im.cpu().numpy()  # torch to numpy

        y = self.session(im)
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

    def preprocess(self, im, imgsz, stride=32, pt=False, fp16=False):
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

        im = im.to(self.device)
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self, preds,
                    img,
                    orig_imgs,
                    # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
                    conf=0.25,
                    iou=0.7,  # (float) intersection over union (IoU) threshold for NMS
                    agnostic_nms=False,  # (bool) class-agnostic NMS
                    max_det=300,  # (int) maximum number of detections per image
                    classes=None,
                    # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
                    ):
        # print("****************************************************")
        # print(f"conf= {conf}")
        # print(f"iou= {iou}")
        # print(f"agnostic_nms= {agnostic_nms}")
        # print(f"max_det= {max_det}")
        # print(f"classes= {classes}")
        # print("****************************************************")
        """Post-processes predictions and returns a list of Results objects."""
        preds = non_max_suppression(preds,
                                    conf_thres=conf,
                                    iou_thres=iou,
                                    agnostic=agnostic_nms,
                                    max_det=max_det,
                                    classes=classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = convert_torch2numpy_batch(orig_imgs)
        # print('#' * 20)
        # print(preds[0].reshape(-1)[:20])

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

    def detect(self, im0: ndarray):
        # return super().detect(im0)
        # im = preprocess([im0])
        im0s = [im0]
        im = self.preprocess(im0s, self.imgsz)

        preds = self.infer(im)
        # print("*" * 20)
        # print(preds.reshape(-1)[:20])

        boxes, confs, cls_ids = self.postprocess(preds, im, im0s)
        # boxes, confs, cls_ids = postprocess(outputs, im.shape[2:], im0.shape[:2], conf=0.25, iou=0.45)
        return boxes, confs, cls_ids

    def predict_image(self, img_path, output_dir="output/", suffix="yolov8_triton_w_torch", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov8_triton_w_torch", save=False):
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
