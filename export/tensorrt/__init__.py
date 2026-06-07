# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: TensorRT 引擎构建子模块

build_fp16_engine 始终可用（trtexec 模式不依赖 tensorrt Python API）。
INT8 构建器使用懒加载——仅在首次调用时导入，避免在无 TensorRT 环境中阻断 import。
"""

__all__ = [
    "build_fp16_engine",
    "build_int8_engine_torch",
    "build_int8_engine_pycuda",
]


def __getattr__(name):
    """惰性导入：避免 python -m export.tensorrt.* 触发 runpy 警告。"""
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
