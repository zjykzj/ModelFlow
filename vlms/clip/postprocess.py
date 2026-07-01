# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : postprocess.py
@Author  : zj
@Description: CLIP 后处理 — embedding 归一化 → similarity → softmax → top-k 排序

CLIP 模型的输出为 image_embed + text_embed，
后处理计算相似度矩阵 → softmax → 排序结果。
"""

from typing import List, Optional

import numpy as np


class CLIPPostprocessor:
    """CLIP 后处理：计算图像与文本 embedding 的相似度并排序。

    Args:
        class_list: 类别/文本名称列表（可选，用于返回 top-k 标签）
        topk: 返回 top-k 结果（默认 5）
    """

    def __init__(self, class_list: Optional[List[str]] = None, topk: int = 5):
        self.class_list = class_list or []
        self.topk = topk

    def __call__(self, image_embed: np.ndarray,
                 text_embed: Optional[np.ndarray] = None) -> dict:
        """计算相似度并返回 top-k 结果。

        Args:
            image_embed: 图像 embedding, shape (dim,) 或 (1, dim)
            text_embed: 文本 embedding, shape (num_texts, dim)，为 None 时仅返回 image_embed

        Returns:
            dict: 包含 similarity, probs, top_indices, top_probs（以及可选的 top_labels）
        """
        if image_embed.ndim > 1:
            image_embed = image_embed[0]

        if text_embed is None:
            return {"image_embed": image_embed}

        if text_embed.ndim == 1:
            text_embed = text_embed[None]

        # 归一化 + 相似度
        image_norm = image_embed / (np.linalg.norm(image_embed) + 1e-10)
        text_norm = text_embed / (np.linalg.norm(text_embed, axis=1, keepdims=True) + 1e-10)

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
