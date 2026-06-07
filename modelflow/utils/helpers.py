# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : helpers.py
@Author  : zj
@Description: 工具函数
"""

import time
from contextlib import ContextDecorator
from typing import Optional


class Profile(ContextDecorator):
    """计时器（可用作装饰器或上下文管理器）

    用法:
        with Profile("inference"):
            result = model(input)
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        label = f" [{self.name}]" if self.name else ""
        print(f"⏱ {label} {self.elapsed * 1000:.1f} ms")
