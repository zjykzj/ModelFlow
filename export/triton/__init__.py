# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: Triton 配置生成子模块
"""

__all__ = [
    "TritonConfigGenerator",
    "ModelRepoBuilder",
]


def __getattr__(name):
    """惰性导入：避免 python -m export.triton.config_generator 触发 runpy 警告。"""
    if name == "TritonConfigGenerator":
        from .config_generator import TritonConfigGenerator
        return TritonConfigGenerator
    if name == "ModelRepoBuilder":
        from .model_repo import ModelRepoBuilder
        return ModelRepoBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
