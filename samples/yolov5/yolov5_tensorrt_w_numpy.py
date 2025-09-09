# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/7 16:22
@File    : yolov5_tensorrt_w_numpy.py
@Author  : zj
@Description: 
"""

from numpy import ndarray
from typing import Union, Tuple, Optional, Any, List

from core.backends.backend_tensorrt import BackendTensorRT
from core.utils.general import Profile
from yolov5_runtime_w_numpy import preprocess, postprocess


class YOLOv5TensorRT:

    def __init__(self, classes, weight: str = 'yolov5s_fp16.engine'):
        super().__init__()
        self.classes = classes
        self.nc = len(classes)

        self.session = BackendTensorRT(weight)
        self.session.load()

        self.input_name = self.session.get_input_names()[0]
        self.net_h, self.net_w = self.session.get_input_shapes()[self.input_name][2:]
        self.output_names = self.session.output_names

    def infer(self, im: ndarray) -> List[Any]:
        output_dict = self.session.infer({self.input_name: im})

        pred = []
        for output_name in self.output_names:
            pred.append(output_dict[output_name])
        return pred

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.45) -> Tuple:
        """
        Detect objects in the image and measure time consumption for each stage.
        Returns:
            boxes, confs, cls_ids, dt
        """
        # Record start time
        dt = (Profile(), Profile(), Profile())

        # --- Preprocessing ---
        with dt[0]:
            im, ratio, padding = preprocess(im0, (self.net_h, self.net_w))
            im_shape = im.shape[2:]  # Model input shape (h, w)
            im0_shape = im0.shape[:2]  # Original image shape (h, w)

        # --- Inference ---
        with dt[1]:
            pred = self.infer(im)

        # --- Postprocessing ---
        with dt[2]:
            boxes, confs, cls_ids = postprocess(
                pred,
                im_shape,
                im0_shape,
                conf=conf,
                iou=iou
            )

        return boxes, confs, cls_ids, dt
