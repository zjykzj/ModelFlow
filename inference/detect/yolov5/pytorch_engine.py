# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:09
@File    : pytorch_engine.py
@Author  : zj
@Description: 
"""

import torch
import numpy as np

from inference.base.engine import BaseInferenceEngine
from inference.engines.pytorch_engine import PyTorchEngine
from inference.detect.preprocessors.numpy_preprocessor import YOLOv5NumpyPreprocessor
from inference.detect.postprocessors.torch_postprocessor import YOLOv5TorchPostprocessor


class YOLOv5PyTorchEngine(BaseInferenceEngine):
    def __init__(self, model_path: str, device: str = 'cuda', img_size: int = 640):
        super().__init__(model_path)
        self.img_size = img_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_wrapper = PyTorchEngine(model_path, device=self.device)
        self.preprocessor = YOLOv5NumpyPreprocessor(img_size=img_size)
        self.postprocessor = YOLOv5TorchPostprocessor()

    def default_preprocessor(self):
        return self.preprocessor

    def default_postprocessor(self):
        return self.postprocessor

    def load_model(self):
        self.model_wrapper.load_model()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        padded = self.preprocessor(image)
        tensor = torch.from_numpy(padded).to(self.model_wrapper.get_device())
        return tensor

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model_wrapper.infer(input_tensor)

    def postprocess(self, outputs: torch.Tensor, original_shape: tuple) -> dict:
        results = self.postprocessor(outputs, original_shape)
        return {
            "boxes": results[0]["boxes"],
            "scores": results[0]["scores"],
            "class_ids": results[0]["class_ids"]
        }

    def __call__(self, image: np.ndarray) -> dict:
        return super().__call__(image)
