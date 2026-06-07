# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: 多模态（CLIP）处理器
"""

from .preprocess import CLIPImagePreprocessor, CLIPTextPreprocessor
from .postprocess import CLIPPostprocessor

__all__ = ["CLIPImagePreprocessor", "CLIPTextPreprocessor", "CLIPPostprocessor"]
