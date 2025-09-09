# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/29 17:38
@File    : yolov5_runtime_w_numpy.py
@Author  : zj
@Description: 
"""

import numpy as np
from numpy import ndarray
from typing import Union, Tuple, Optional, Any, List

from core.backends.backend_runtime import BackendRuntime
from core.utils.general import Profile
from core.utils.v5.preprocessor import letterbox
from core.utils.v5.postprocessor import non_max_suppression, scale_boxes


def preprocess(im0: ndarray, img_size: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False,
               fp16: bool = False) -> Tuple:
    """
    Preprocess the input image: resize with padding, convert format from HWC to CHW and BGR to RGB,
    normalize values from range 0-255 to 0.0-1.0, and add a batch dimension.

    Parameters:
    - im0: Input image as an ndarray.
    - img_size: Target image size for resizing (can be an integer or a tuple).
    - stride: Stride value for resizing operations.
    - auto: Parameter for letterbox resizing that determines whether to automatically adjust dimensions.
    - fp16: If True, converts the image data type to float16; otherwise, uses float32.

    Returns:
    - im: Preprocessed image ready for input into a neural network model.
    - ratio: The ratio of resized image dimensions relative to the original.
    - (dw, dh): Padding widths on the x and y axes added during preprocessing.
    """
    im, ratio, (dw, dh) = letterbox(im0, img_size, stride=stride, auto=auto)  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = im.astype(np.float16) if fp16 else im.astype(np.float32)  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    return im, ratio, (dw, dh)


def postprocess(
        pred: List[ndarray],
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


class YOLOv5RuntimeNumpy:

    def __init__(self, weight: str = 'yolov5s.onnx', providers=None):
        super().__init__()
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = BackendRuntime(weight, providers=providers)
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
