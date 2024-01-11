# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/11 15:33
@File    : yolov8.py
@Author  : zj
@Description: 
"""

import torch

import numpy as np

from checks import check_imgsz

import os
from general import LOGGER

import time
import torchvision

from yolov8_util import LetterBox
from torch_util import non_max_suppression, convert_torch2numpy_batch, scale_boxes


def pre_transform(im, imgsz, stride=32, pt=False):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = all(x.shape == im[0].shape for x in im)
    print(f"imgsz: {imgsz}")
    print(f"auto = {same_shapes and pt}")
    print(f"stride = {stride}")
    letterbox = LetterBox(imgsz, auto=same_shapes and pt, stride=stride)
    return [letterbox(image=x) for x in im]


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


def load_onnx(weight: str):
    assert os.path.isfile(weight), weight

    LOGGER.info(f'Loading {weight} for ONNX Runtime inference...')
    import onnxruntime
    providers = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(weight, providers=providers)
    output_names = [x.name for x in session.get_outputs()]
    metadata = session.get_modelmeta().custom_metadata_map  # metadata
    LOGGER.info(f"metadata: {metadata}")

    session = session
    output_names = output_names
    dtype = np.float32
    LOGGER.info(f"Init Done. Work with {dtype}")

    return session, output_names, dtype


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
    return preds


if __name__ == '__main__':
    imgsz = 640
    stride = 32
    imgsz = check_imgsz(imgsz, stride=stride, min_dim=2)
    print(imgsz)

    session, output_names, dtype = load_onnx("yolov8n.onnx")

    import cv2

    path = "../../assets/bus.jpg"
    im0 = cv2.imread(path)  # BGR
    if im0 is None:
        raise FileNotFoundError(f'Image Not Found {path}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im0s = [im0]
    im = preprocess(im0s, imgsz, device)
    print(im.shape)

    preds = session.run(output_names, {session.get_inputs()[0].name: im.numpy()})[0]
    # print(len(preds), preds[0].shape)

    preds = torch.from_numpy(preds)
    print(preds.shape)
    preds = postprocess(preds, im, im0s)
    print(len(preds), preds[0].shape)
