# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: Triton configuration generation submodule
"""

__all__ = [
    "TritonConfigGenerator",
    "ModelRepoBuilder",
]


def __getattr__(name):
    """Lazy import: avoid runpy warnings when running python -m export.triton.config_generator."""
    if name == "TritonConfigGenerator":
        from .config_generator import TritonConfigGenerator
        return TritonConfigGenerator
    if name == "ModelRepoBuilder":
        from .model_repo import ModelRepoBuilder
        return ModelRepoBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
