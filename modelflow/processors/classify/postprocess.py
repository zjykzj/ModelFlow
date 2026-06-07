# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : postprocess.py
@Author  : zj
@Description: 分类后处理 — softmax → top-k

用法:
    postprocessor = ClassifyPostprocessor(topk=5)
    result = postprocessor(raw_outputs)
    # result = {"class_ids": [3, 7, ...], "scores": [0.9, 0.05, ...], "class_names": [...]}
"""

from typing import List, Optional

import numpy as np

from modelflow.core.interfaces import BasePostprocessor
from modelflow.core.registry import PROCESSORS


@PROCESSORS.register("classify_postprocess_npy")
class ClassifyPostprocessor(BasePostprocessor):
    """分类后处理

    Args:
        topk: 返回 top-k 结果（默认 5）
        class_list: 类别名称列表（可选，用于返回 class_names）
    """

    def __init__(self, topk: int = 5, class_list: Optional[List[str]] = None):
        self.topk = topk
        self.class_list = class_list

    def __call__(self, raw: List[np.ndarray], **kwargs) -> dict:
        logits = raw[0]  # (1, num_classes) or (num_classes,)

        if logits.ndim == 2:
            logits = logits[0]

        # softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        # top-k
        topk = min(self.topk, len(probs))
        indices = np.argpartition(probs, -topk)[-topk:]
        indices = indices[np.argsort(-probs[indices])]

        result = {
            "class_ids": indices.tolist(),
            "scores": probs[indices].tolist(),
        }

        if self.class_list is not None:
            result["class_names"] = [self.class_list[i] if i < len(self.class_list) else str(i)
                                     for i in indices]

        return result
