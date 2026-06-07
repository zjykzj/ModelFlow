# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: 可视化模块

检测/分割的可视化委托 DataFlow-CV 实现。
分类/语义分割的可视化本地实现。
"""

from .detect import DetectVisualizer

__all__ = ["DetectVisualizer"]
