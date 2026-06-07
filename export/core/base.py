# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : base.py
@Author  : zj
@Description: BaseExporter — 所有导出器的统一抽象基类

所有导出器（torchvision / Ultralytics / 自定义）继承此类，
统一 export_onnx 接口和验证流程。
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseExporter(ABC):
    """导出器抽象基类"""

    def __init__(self, model_name: str, opset: int = 12):
        self.model_name = model_name
        self.opset = opset

    @abstractmethod
    def export_onnx(self, output_path: str, **kwargs) -> str:
        """导出 ONNX 模型

        Args:
            output_path: ONNX 文件保存路径
            **kwargs: 各导出器特有参数

        Returns:
            str: ONNX 文件的绝对路径
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r}, opset={self.opset})"
