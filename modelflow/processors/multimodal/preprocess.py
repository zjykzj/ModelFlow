# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : preprocess.py
@Author  : zj
@Description: 多模态预处理 — CLIP 图像/文本预处理

图像流程: OpenCV BGR → RGB → Resize(224) → CenterCrop(224) → Normalize(CLIP stats) → CHW
文本流程: 文本 → CLIP tokenizer → token IDs
"""

from typing import List, Optional, Union

import cv2
import numpy as np

from modelflow.core.interfaces import BasePreprocessor
from modelflow.core.registry import PROCESSORS

# CLIP 标准 Normalize 参数
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


@PROCESSORS.register("multimodal_image_preprocess_npy")
class CLIPImagePreprocessor(BasePreprocessor):
    """CLIP 图像预处理（NumPy 实现）

    Args:
        input_size: 输入尺寸（默认 224）
    """

    def __init__(self, input_size: int = 224):
        self.input_size = input_size

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        # Resize 短边到 input_size
        if h < w:
            new_h = self.input_size
            new_w = int(round(w * self.input_size / h))
        else:
            new_w = self.input_size
            new_h = int(round(h * self.input_size / w))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # CenterCrop
        ch, cw = resized.shape[:2]
        sh = (ch - self.input_size) // 2
        sw = (cw - self.input_size) // 2
        cropped = resized[sh:sh + self.input_size, sw:sw + self.input_size]

        # CLIP Normalize
        normalized = cropped.astype(np.float32) / 255.0
        normalized = (normalized - CLIP_MEAN) / CLIP_STD

        chw = np.transpose(normalized, (2, 0, 1))
        return chw[None]


@PROCESSORS.register("multimodal_text_preprocess_npy")
class CLIPTextPreprocessor(BasePreprocessor):
    """CLIP 文本预处理（需要 CLIP 库）

    对文本列表进行 tokenize，输出 token IDs。
    需要安装 CLIP 库: pip install git+https://github.com/openai/CLIP.git
    """

    def __init__(self, max_length: int = 77):
        self.max_length = max_length
        self._clip = None

    @property
    def clip(self):
        if self._clip is None:
            try:
                import clip
                self._clip = clip
            except ImportError:
                raise ImportError(
                    "CLIP not installed. Install: "
                    "pip install git+https://github.com/openai/CLIP.git"
                )
        return self._clip

    def __call__(self, text: Union[str, List[str]], **kwargs) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        tokens = self.clip.tokenize(text, truncate=True).numpy()
        return tokens
