# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : postprocess.py
@Author  : zj
@Description: 多模态后处理 — similarity → probability → ranking

CLIP 模型的输出为 image_embed + text_embed，
后处理计算相似度矩阵 → softmax → 排序结果。
"""

from typing import List

import numpy as np

from modelflow.core.interfaces import BasePostprocessor
from modelflow.core.registry import PROCESSORS


@PROCESSORS.register("multimodal_postprocess_npy")
class CLIPPostprocessor(BasePostprocessor):
    """CLIP 后处理

    Args:
        class_list: 类别/文本名称列表（可选）
        topk: 返回 top-k 结果
    """

    def __init__(self, class_list=None, topk: int = 5):
        self.class_list = class_list or []
        self.topk = topk

    def __call__(self, raw: List[np.ndarray], **kwargs) -> dict:
        # raw[0]: image_embed (1, dim) or (dim,)
        # raw[1]: text_embed (num_texts, dim) or None（需要外部提供）
        image_embed = raw[0]
        text_embed = raw[1] if len(raw) > 1 else None

        if image_embed.ndim > 1:
            image_embed = image_embed[0]

        if text_embed is None and "text_embed" in kwargs:
            text_embed = kwargs["text_embed"]
            if text_embed.ndim > 1:
                text_embed = text_embed[0] if len(text_embed) == 1 else text_embed

        if text_embed is None:
            return {"image_embed": image_embed}

        # 归一化 + 相似度
        image_norm = image_embed / (np.linalg.norm(image_embed) + 1e-10)
        text_norm = text_embed / (np.linalg.norm(text_embed, axis=1, keepdims=True) + 1e-10)

        if text_norm.ndim == 1:
            text_norm = text_norm[None]

        similarity = image_norm @ text_norm.T  # (num_texts,)
        probs = np.exp(similarity - similarity.max())
        probs = probs / probs.sum()

        # top-k
        topk = min(self.topk, len(probs))
        indices = np.argsort(-probs)[:topk]

        result = {
            "similarity": similarity.tolist(),
            "probs": probs.tolist(),
            "top_indices": indices.tolist(),
            "top_probs": probs[indices].tolist(),
        }

        if self.class_list:
            result["top_labels"] = [self.class_list[i] for i in indices]

        return result
