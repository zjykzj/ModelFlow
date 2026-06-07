# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: ONNX 导出子模块
"""

__all__ = [
    "TorchvisionExporter",
    "UltralyticsExporter",
    "optimize_onnx",
]


def __getattr__(name):
    """惰性导入：避免 python -m export.onnx.ultralytics 触发 runpy 警告。"""
    if name == "TorchvisionExporter":
        from .convert import TorchvisionExporter
        return TorchvisionExporter
    if name == "UltralyticsExporter":
        from .ultralytics import UltralyticsExporter
        return UltralyticsExporter
    if name == "optimize_onnx":
        from .optimize import optimize_onnx
        return optimize_onnx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
