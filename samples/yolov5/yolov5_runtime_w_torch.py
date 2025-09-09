# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/31 17:38
@File    : yolov5_runtime_w_torch.py
@Author  : zj
@Description: 
"""

from typing import Optional, Union, Tuple, Any, List

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from core.utils.v5.preprocessor import letterbox
from core.utils.general import Profile
from core.backends.backend_runtime import BackendRuntime
from torch_util import non_max_suppression, scale_boxes


def preprocess(im0: ndarray, img_size: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False,
               fp16: bool = False) -> Tuple:
    """
    Sourced from

    1. https://github.com/ultralytics/yolov5/blob/915bbf294bb74c859f0b41f1c23bc395014ea679/utils/dataloaders.py#L231
    2. https://github.com/ultralytics/yolov5/blob/915bbf294bb74c859f0b41f1c23bc395014ea679/detect.py#L117
    """
    im, ratio, (dw, dh) = letterbox(im0, img_size, stride=stride, auto=auto)  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im)
    im = im.half() if fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    return im.numpy(), ratio, (dw, dh)


def postprocess(
        pred: Union[Tensor, List[Tensor]],
        im_shape: Tuple,  # (h, w) of input to model
        im0_shape: Tuple,  # (h, w) of original image
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[list] = None,
        agnostic: bool = False,
        max_det: int = 300,
) -> Tuple:
    """
     Postprocessing: NMS + coordinate scaling.
     Returns:
         boxes, confs, cls_ids
     """
    det = non_max_suppression(pred, conf, iou, classes, agnostic, max_det=max_det)[0]

    if len(det) > 0:
        boxes = scale_boxes(im_shape, det[:, :4], im0_shape).round()
        confs = det[:, 4:5]
        cls_ids = det[:, 5:6]
    else:
        boxes, confs, cls_ids = [], [], []
    return boxes, confs, cls_ids


class YOLOv5RuntimeTorch:

    def __init__(self, classes, weight: str = 'yolov5s.onnx', providers: List[str] = None):
        super().__init__()
        self.classes = classes
        self.nc = len(classes)

        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = BackendRuntime(weight, providers=providers)
        self.session.load()

        self.input_name = self.session.get_input_names()[0]
        self.net_h, self.net_w = self.session.get_input_shapes()[self.input_name][2:]
        self.output_names = self.session.output_names

        self.device = torch.device('cpu')

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def infer(self, im: ndarray) -> List[Any]:
        output_dict = self.session.infer({self.input_name: im})

        pred = []
        for output_name in self.output_names:
            pred.append(self.from_numpy(output_dict[output_name]))
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
