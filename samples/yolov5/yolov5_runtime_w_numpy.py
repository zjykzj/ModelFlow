# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/29 17:38
@File    : yolov5_runtime_w_numpy.py
@Author  : zj
@Description: 
"""

import time
import logging

import numpy as np
from numpy import ndarray
from typing import Union, Tuple, Optional

from core.backends.backend_runtime import BackendRuntime
from numpy_util import letterbox, non_max_suppression, scale_boxes


def preprocess(im0: ndarray, img_size: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False) -> ndarray:
    """
    图像预处理：缩放、转格式、归一化、添加 batch 维度。
    """
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255.0  # 0-255 to 0.0-1.0
    if im.ndim == 3:
        im = im[None]  # expand for batch dim
    return im


def postprocess(
        preds: ndarray,
        im_shape: tuple,  # (h, w) of input to model
        im0_shape: tuple,  # (h, w) of original image
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[list] = None,
        agnostic: bool = False,
        max_det: int = 300,
) -> tuple:
    """
    后处理：NMS + 坐标缩放。
    Returns:
        boxes, confs, cls_ids
    """
    pred = non_max_suppression(preds, conf, iou, classes, agnostic, max_det=max_det)[0]
    boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
    confs = pred[:, 4:5]
    cls_ids = pred[:, 5:6]
    return boxes, confs, cls_ids


class YOLOv5Runtime:

    def __init__(self, weight: str = 'yolov5s.onnx'):
        super().__init__()
        self.session = BackendRuntime(weight)
        self.session.load()

        input_name = self.session.get_input_names()[0]
        self.net_h, self.net_w = self.session.get_input_shapes()[input_name][2:]

    def infer(self, im: ndarray) -> ndarray:
        input_name = self.session.input_names[0]
        output_dict = self.session.infer({input_name: im})
        output_name = self.session.output_names[0]
        return output_dict[output_name]

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.45) -> tuple:
        """
        Detect objects in the image and measure time consumption for each stage.
        Returns:
            boxes, confs, cls_ids
        """
        # Record start time
        t0 = time.perf_counter()

        # --- Preprocessing ---
        t_pre_start = time.perf_counter()
        im = preprocess(im0, (self.net_h, self.net_w))
        t_pre_end = time.perf_counter()

        # --- Inference ---
        t_inf_start = time.perf_counter()
        outputs = self.infer(im)
        t_inf_end = time.perf_counter()

        # --- Postprocessing ---
        t_post_start = time.perf_counter()
        boxes, confs, cls_ids = postprocess(
            outputs,
            im.shape[2:],  # Model input shape (h, w)
            im0.shape[:2],  # Original image shape (h, w)
            conf=conf,
            iou=iou
        )
        t_post_end = time.perf_counter()

        # --- Timing statistics ---
        pre_time = (t_pre_end - t_pre_start) * 1000  # ms
        inf_time = (t_inf_end - t_inf_start) * 1000
        post_time = (t_post_end - t_post_start) * 1000
        total_time = (t_post_end - t0) * 1000

        logging.info(
            f"Detect time - Pre: {pre_time:.2f}ms | "
            f"Infer: {inf_time:.2f}ms | "
            f"Post: {post_time:.2f}ms | "
            f"Total: {total_time:.2f}ms"
        )

        return boxes, confs, cls_ids
