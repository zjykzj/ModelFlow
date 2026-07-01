# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : coco.py
@Author  : zj
@Description: COCO 数据集 — 支持检测和实例分割任务

加载 COCO 格式图片目录，与标注 JSON 配对用于评估。
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from data.base import BaseDataset


class COCODataset(BaseDataset):
    """COCO 数据集（检测 / 实例分割）

    加载图片目录中的所有图片，可通过 anno_json 指定标注文件用于评估。

    Args:
        img_dir: 图片目录（如 val2017/）
        class_list: 类别名称列表
        task: 任务类型 — "detect" 或 "segment"
        anno_json: COCO JSON 标注路径（用于评估）
        img_size: 目标尺寸（保留字段）
    """

    def __init__(
        self,
        img_dir: str,
        class_list: List[str],
        task: str = "detect",
        anno_json: Optional[str] = None,
        img_size: int = 640,
    ):
        self.img_dir = img_dir
        self.class_list = class_list
        self.task = task
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
            "task": self.task,
        }
        return image, gt

    def get_gt_json(self) -> str:
        return self._anno_json or ""
