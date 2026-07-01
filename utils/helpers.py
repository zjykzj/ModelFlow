# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : helpers.py
@Author  : zj
@Description: 通用辅助工具 — Profile 计时器
"""

import time
from contextlib import ContextDecorator


class Profile(ContextDecorator):
    """Profile 计时器（上下文管理器 + 装饰器一体化）

    Usage:
        with Profile("inference") as p:
            result = model(input)

        @Profile("training")
        def train():
            ...
    """

    def __init__(self, name: str = "Profile"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start_time
        if elapsed >= 1.0:
            print(f"[{self.name}] Elapsed: {elapsed:.2f} s")
        else:
            print(f"[{self.name}] Elapsed: {elapsed * 1000:.2f} ms")
        return False
