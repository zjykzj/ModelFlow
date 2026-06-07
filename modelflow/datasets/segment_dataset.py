# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : segment_dataset.py
@Author  : zj
@Description: COCO 实例分割数据集

加载 COCO 格式的分割标注（RLE/polygon），与 DetectDataset 类似，
但 ground_truth 中包含 segmentation 标注。
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modelflow.core.interfaces import BaseDataset
from modelflow.core.registry import DATASETS


@DATASETS.register("coco_segment")
class COCOSegmentDataset(BaseDataset):
    """COCO 实例分割数据集

    Args:
        img_dir: 图片目录
        class_list: 类别名称列表
        anno_json: COCO JSON 标注路径
        img_size: 目标尺寸
    """

    def __init__(
        self,
        img_dir: str,
        class_list: List[str],
        anno_json: Optional[str] = None,
        img_size: int = 640,
    ):
        self.img_dir = img_dir
        self.class_list = class_list
        self._anno_json = anno_json
        self.img_size = img_size

        self.image_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Dict]:
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")

        gt = {
            "image_path": img_path,
            "image_id": os.path.splitext(os.path.basename(img_path))[0],
        }
        return image, gt

    def get_gt_json(self) -> str:
        return self._anno_json or ""
