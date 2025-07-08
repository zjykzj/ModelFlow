# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:03
@File    : engine.py
@Author  : zj
@Description: 统一引擎接口
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union
import numpy as np


class InferenceError(Exception):
    """Base class for inference-related exceptions"""
    pass


class ModelLoadError(InferenceError):
    """Raised when model loading fails"""
    pass


class BaseInferenceEngine(ABC):

    def __init__(
            self,
            model_path: str,
            device: str = "cpu",
            preprocessor: Optional[Callable] = None,
            postprocessor: Optional[Callable] = None
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    def preprocess(self, image: np.ndarray) -> Any:
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not set.")
        return self.preprocessor(image)

    def postprocess(self, outputs: Any, original_shape: tuple) -> Any:
        if self.postprocessor is None:
            raise ValueError("Postprocessor is not set.")
        return self.postprocessor(outputs, original_shape)

    @abstractmethod
    def infer(self, input_tensor: Any) -> Any:
        """Perform inference on the input tensor(s)."""
        pass

    def __call__(self, image: np.ndarray) -> Any:
        input_tensor = self.preprocess(image)
        raw_output = self.infer(input_tensor)
        return self.postprocess(raw_output, image.shape)
