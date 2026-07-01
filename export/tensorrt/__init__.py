# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: TensorRT engine building submodule

build_fp16_engine is always available (trtexec mode does not depend on the tensorrt Python API).
INT8 builders use lazy loading — imported only on first call to avoid blocking imports in environments without TensorRT.
"""

__all__ = [
    "build_fp16_engine",
    "build_int8_engine_torch",
    "build_int8_engine_pycuda",
]


def __getattr__(name):
    """Lazy import: avoids triggering runpy warnings from python -m export.tensorrt.*."""
    if name == "build_fp16_engine":
        from .build_fp16 import build_fp16_engine
        return build_fp16_engine
    if name == "build_int8_engine_torch":
        from .build_int8 import build_int8_engine_torch
        return build_int8_engine_torch
    if name == "build_int8_engine_pycuda":
        from .build_int8_pycuda import build_int8_engine_pycuda
        return build_int8_engine_pycuda
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
