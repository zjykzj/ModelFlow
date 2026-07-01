# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : postprocess.py
@Author  : zj
@Description: 语义分割后处理 — argmax

流程: logits (N, num_classes, H, W) → argmax → class_id map
"""

from typing import List

import numpy as np

from modelflow.interfaces import BasePostprocessor


class SemanticSegPostprocessor(BasePostprocessor):
    """语义分割后处理（NumPy 实现）

    Returns:
        class_map: (H, W) uint8 class_id map
        num_classes: number of classes in logits
    """

    def __call__(self, raw: List[np.ndarray], **kwargs) -> dict:
        logits = raw[0]  # (1, num_classes, H, W) or (num_classes, H, W)

        if logits.ndim == 4:
            logits = logits[0]  # (num_classes, H, W)

        # argmax → class_id map
        class_map = np.argmax(logits, axis=0).astype(np.uint8)  # (H, W)

        result = {
            "class_map": class_map,           # (H, W) uint8
            "num_classes": logits.shape[0],
        }

        return result
