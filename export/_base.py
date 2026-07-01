# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : base.py
@Author  : zj
@Description: BaseExporter — unified abstract base class for all exporters

All exporters (torchvision / Ultralytics / custom) inherit from this class,
sharing a unified export_onnx interface and validation pipeline.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseExporter(ABC):
    """Abstract base class for exporters"""

    def __init__(self, model_name: str, opset: int = 12):
        self.model_name = model_name
        self.opset = opset

    @abstractmethod
    def export_onnx(self, output_path: str, **kwargs) -> str:
        """Export an ONNX model

        Args:
            output_path: File path to save the ONNX model
            **kwargs: Exporter-specific parameters

        Returns:
            str: Absolute path to the exported ONNX file
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, opset={self.opset})"
