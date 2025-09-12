# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/7 16:22
@File    : yolov5_tensorrt_w_numpy.py
@Author  : zj
@Description: 
"""

from numpy import ndarray
from typing import Union, Tuple, Optional, Any, List

from core.backends.backend_triton import BackendTriton
from core.utils.general import Profile
from yolov5_runtime_w_numpy import preprocess, postprocess


class YOLOv5Triton:

    def __init__(self, classes, weight: str = 'DET_YOLOv5s_ONNX'):
        super().__init__()
        self.classes = classes
        self.nc = len(classes)

        self.session = BackendTriton(weight, server_url="localhost:8001", protocol='grpc')
        self.session.load()

        self.input_name = self.session.get_input_names()[0]
        self.input_type = self.session.get_input_dtypes()[self.input_name]
        self.net_h, self.net_w = self.session.get_input_shapes()[self.input_name][2:]
        self.output_names = self.session.output_names

        self.fp16 = True if isinstance(self.input_type, np.float16) else False

    def infer(self, im: ndarray) -> List[Any]:
        output_dict = self.session.infer({self.input_name: im})

        pred = []
        for output_name in self.output_names:
            pred.append(output_dict[output_name].copy())  # fix ValueError: assignment destination is read-only
        return pred

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.45) -> Tuple[ndarray, ndarray, ndarray, Tuple]:
        """
        Detect objects in the image and measure time consumption for each stage.
        Returns:
            boxes, confs, cls_ids, dt
        """
        # Record start time
        dt = (Profile(), Profile(), Profile())

        # --- Preprocessing ---
        with dt[0]:
            im, ratio, padding = preprocess(im0, (self.net_h, self.net_w), fp16=self.fp16)
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
