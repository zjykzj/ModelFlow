# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/7 16:22
@File    : yolov5_tensorrt_w_numpy.py
@Author  : zj
@Description: 
"""

import logging

import numpy as np
from numpy import ndarray
from typing import Union, Tuple, Optional, Any, List

from core.backends.backend_tensorrt import BackendTensorRT
from core.utils.general import Profile
from yolov5_runtime_w_numpy import preprocess, postprocess


class YOLOv5TensorRT:

    def __init__(self, weight: str = 'yolov5s.onnx'):
        super().__init__()
        self.session = BackendTensorRT(weight)
        self.session.load()

        self.input_name = self.session.get_input_names()[0]
        self.net_h, self.net_w = self.session.get_input_shapes()[self.input_name][2:]
        self.output_names = self.session.output_names

    def infer(self, im: ndarray) -> list[Any]:
        output_dict = self.session.infer({self.input_name: im})

        pred = []
        for output_name in self.output_names:
            pred.append(output_dict[output_name])
        return pred

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.45, fp16: bool = False) -> tuple:
        """
        Detect objects in the image and measure time consumption for each stage.
        Returns:
            boxes, confs, cls_ids
        """
        # Record start time
        dt = (Profile(), Profile(), Profile())

        # --- Preprocessing ---
        with dt[0]:
            im, ratio, padding = preprocess(im0, (self.net_h, self.net_w), fp16=fp16)

        # --- Inference ---
        with dt[1]:
            pred = self.infer(im)

        # --- Postprocessing ---
        with dt[2]:
            boxes, confs, cls_ids = postprocess(
                pred,
                im.shape[2:],  # Model input shape (h, w)
                im0.shape[:2],  # Original image shape (h, w)
                conf=conf,
                iou=iou
            )

        # --- Timing statistics ---
        pre_time = dt[0].t * 1000  # ms
        inf_time = dt[1].t * 1000
        post_time = dt[2].t * 1000
        total_time = sum([t.t for t in dt]) * 1000

        logging.info(
            f"Detect time - Pre: {pre_time:.2f}ms | "
            f"Infer: {inf_time:.2f}ms | "
            f"Post: {post_time:.2f}ms | "
            f"Total: {total_time:.2f}ms"
        )

        return boxes, confs, cls_ids
