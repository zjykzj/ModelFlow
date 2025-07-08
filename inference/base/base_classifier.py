# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:04
@File    : base_classifier.py
@Author  : zj
@Description: 
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

from .utils import load_class_names, softmax


class BaseClassifier:
    DEFAULT_CLASS_NAMES = "imagenet.names"

    def __init__(
            self,
            model_path: str,
            device: str = "cpu",
            preprocessor: Optional[object] = None,
            postprocessor: Optional[object] = None,
            top_k: int = 5,
            class_names: Optional[List[str]] = None
    ):
        self.model_path = model_path
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.top_k = top_k
        self.class_names = class_names or load_class_names(self.DEFAULT_CLASS_NAMES)

    def get_top_k(self, outputs: np.ndarray) -> List[Tuple[int, float]]:
        probs = self.softmax(outputs)
        top_k_indices = np.argsort(probs)[-self.top_k:][::-1]
        return [(i, probs[i]) for i in top_k_indices]

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def draw_classification(self, image: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        predictions = result.get("predictions", [])
        x, y = 20, 40

        for i, (cls_id, score) in enumerate(predictions):
            label = f"{self.class_names[cls_id]}: {score:.2%}"
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30

        return image
