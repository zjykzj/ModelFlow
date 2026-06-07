# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : coco_detection.py
@Author  : zj
@Description: COCO 检测数据集

加载 COCO 格式的检测数据集，支持图片 + 标注配对。
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modelflow.core.interfaces import BaseDataset
from modelflow.core.registry import DATASETS


@DATASETS.register("coco_detection")
class COCODetectionDataset(BaseDataset):
    """COCO 检测数据集

    Args:
        img_dir: 图片目录（如 val2017/）
        label_dir: YOLO 格式标签目录（可选）
        anno_json: COCO JSON 标注路径（用于评估）
        class_list: 类别名称列表
        img_size: 目标尺寸（影响 label 中的 bbox 缩放）
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

        # 扫描图片
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

        gt = {"image_path": img_path, "image_id": os.path.splitext(os.path.basename(img_path))[0]}
        return image, gt

    def get_gt_json(self) -> str:
        return self._anno_json or ""
