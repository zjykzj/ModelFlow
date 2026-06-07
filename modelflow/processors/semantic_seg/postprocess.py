# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : postprocess.py
@Author  : zj
@Description: 语义分割后处理 — argmax + colormap

流程: logits (N, num_classes, H, W) → argmax → class_id map → colormap (可选)
"""

from typing import List, Optional, Tuple

import numpy as np

from modelflow.core.interfaces import BasePostprocessor
from modelflow.core.registry import PROCESSORS


def _cityscapes_colormap() -> np.ndarray:
    """Cityscapes 标准 colormap (19 + 1 类)"""
    cmap = np.zeros((20, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]        # unlabeled
    cmap[1] = [128, 64, 128]   # road
    cmap[2] = [244, 35, 232]   # sidewalk
    cmap[3] = [70, 70, 70]     # building
    cmap[4] = [102, 102, 156]  # wall
    cmap[5] = [190, 153, 153]  # fence
    cmap[6] = [153, 153, 153]  # pole
    cmap[7] = [250, 170, 30]   # traffic light
    cmap[8] = [220, 220, 0]    # traffic sign
    cmap[9] = [107, 142, 35]   # vegetation
    cmap[10] = [152, 251, 152] # terrain
    cmap[11] = [70, 130, 180]  # sky
    cmap[12] = [220, 20, 60]   # person
    cmap[13] = [255, 0, 0]     # rider
    cmap[14] = [0, 0, 142]     # car
    cmap[15] = [0, 0, 70]      # truck
    cmap[16] = [0, 60, 100]    # bus
    cmap[17] = [0, 80, 100]    # train
    cmap[18] = [0, 254, 206]   # motorcycle
    cmap[19] = [0, 128, 0]     # bicycle
    return cmap


def _voc_colormap(num_classes: int = 21) -> np.ndarray:
    """PASCAL VOC 标准 colormap"""
    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r, g, b = 0, 0, 0
        j = i
        for k in range(8):
            r |= ((j >> 0) & 1) << (7 - k)
            g |= ((j >> 1) & 1) << (7 - k)
            b |= ((j >> 2) & 1) << (7 - k)
            j >>= 3
        cmap[i] = [r, g, b]
    return cmap


def get_colormap(name: str = "voc", num_classes: int = 21) -> np.ndarray:
    """获取 colormap

    Args:
        name: "voc" 或 "cityscapes"
        num_classes: 类别数（仅 voc 使用）

    Returns:
        (num_classes, 3) uint8 colormap
    """
    if name == "cityscapes":
        return _cityscapes_colormap()
    return _voc_colormap(num_classes)


@PROCESSORS.register("semantic_seg_postprocess_npy")
class SemanticSegPostprocessor(BasePostprocessor):
    """语义分割后处理（NumPy 实现）

    Args:
        apply_colormap: 是否将 class_id map 转为彩色掩码（默认 True）
        colormap: colormap 名称 ("voc" / "cityscapes") 或自定义 (N,3) 数组
        num_classes: 类别数（仅 voc 默认模式使用）
    """

    def __init__(
        self,
        apply_colormap: bool = True,
        colormap: Optional[str] = None,
        num_classes: int = 21,
    ):
        self.apply_colormap = apply_colormap
        self.colormap_name = colormap or "voc"
        self.num_classes = num_classes

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

        # 可选：生成彩色掩码
        if self.apply_colormap:
            cmap = get_colormap(self.colormap_name, self.num_classes)
            colormap = cmap[class_map]  # (H, W, 3)
            result["colormap"] = colormap

        return result
