# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : ops.py
@Author  : zj
@Description: 通用数学/几何算子 — 项目通用
"""

import numpy as np


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """(x_center, y_center, w, h) → (x1, y1, x2, y2)"""
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y
