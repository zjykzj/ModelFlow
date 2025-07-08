# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:06
@File    : base_detector.py
@Author  : zj
@Description: 
"""

import os
from abc import ABC

import cv2

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from .engine import BaseInferenceEngine
from .utils import load_class_names


class BaseDetector(BaseInferenceEngine, ABC):
    DEFAULT_CLASS_NAMES = "coco.names"

    def __init__(
            self,
            model_path: str,
            device: str = "cpu",
            preprocessor: Optional[Callable] = None,
            postprocessor: Optional[Callable] = None,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            class_names: Optional[List[str]] = None
    ):
        super().__init__(model_path=model_path, device=device, preprocessor=preprocessor, postprocessor=postprocessor)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_names = class_names or load_class_names(self.DEFAULT_CLASS_NAMES)

    @staticmethod
    def load_class_names(dataset_name: str = "coco") -> List[str]:
        """
        加载数据集类别名称，默认支持 coco / imagenet 等
        """
        class_file = {
            "coco": os.path.join(os.path.dirname(__file__), "..", "..", "assets", "coco", "coco.names"),
            "imagenet": os.path.join(os.path.dirname(__file__), "..", "..", "assets", "imagenet", "imagenet.names")
        }.get(dataset_name.lower())

        if not class_file or not os.path.isfile(class_file):
            raise FileNotFoundError(f"Class name file for {dataset_name} not found.")

        with open(class_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def draw_detections(self, image: np.ndarray, result: Dict[str, np.ndarray]) -> np.ndarray:
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        class_ids = result.get("class_ids", [])

        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            cls_id = class_ids[i]

            label = f"{self.class_names[cls_id]} {score:.2f}"
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image
