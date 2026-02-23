# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 17:18
@File    : helpers.py
@Author  : zj
@Description: 
"""

import time
from contextlib import ContextDecorator


class Profile(ContextDecorator):
    """
    A class to measure the execution time of a block of code or function.
    Can be used as a decorator or as a context manager.

    Example usage:
        @Profile()
        def my_function():
            # Your code here

        with Profile() as prof:
            # Your code here
        print(f"Execution time: {prof.dt:.2f} seconds")
    """

    def __init__(self, name=None):
        self.name = name  # Optional name for the profiled section
        self.start_time = None
        self.end_time = None
        self.dt = 0.0  # Delta time (execution duration)

    def __enter__(self):
        self.start_time = time.time()  # Record start time
        return self

    def __exit__(self, *exc):
        self.end_time = time.time()  # Record end time
        self.dt = self.end_time - self.start_time  # Calculate delta time
        if self.name:
            print(f"[{self.name}] Execution time: {self.dt:.2f} seconds")
        return False
