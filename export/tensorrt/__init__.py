# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : __init__.py
@Author  : zj
@Description: TensorRT 引擎构建子模块

build_fp16_engine 始终可用（trtexec 模式不依赖 tensorrt Python API）。
INT8 构建器使用懒加载——仅在首次调用时导入，避免在无 TensorRT 环境中阻断 import。
"""

from .build_fp16 import build_fp16_engine


def build_int8_engine_torch(*args, **kwargs):
    """PyTorch 校准器 INT8 引擎构建（懒加载）

    首次调用时导入 build_int8 模块；如 TensorRT 未安装，会给出安装指引。
    """
    from .build_int8 import build_int8_engine_torch as _impl
    return _impl(*args, **kwargs)


def build_int8_engine_pycuda(*args, **kwargs):
    """PyCUDA 校准器 INT8 引擎构建（懒加载）

    首次调用时导入 build_int8_pycuda 模块；如 TensorRT 未安装，会给出安装指引。
    """
    from .build_int8_pycuda import build_int8_engine_pycuda as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "build_fp16_engine",
    "build_int8_engine_torch",
    "build_int8_engine_pycuda",
]
