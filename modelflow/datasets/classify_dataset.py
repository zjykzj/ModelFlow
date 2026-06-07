# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : classify_dataset.py
@Author  : zj
@Description: 分类数据集

按目录结构组织：root/class_name/*.jpg
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modelflow.core.interfaces import BaseDataset
from modelflow.core.registry import DATASETS


@DATASETS.register("classify_dataset")
class ClassifyDataset(BaseDataset):
    """分类数据集

    目录结构：
        root/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img3.jpg

    Args:
        root: 数据集根目录（子目录名为类别名）
        class_list: 类别名称列表（按索引匹配）
    """

    def __init__(
        self,
        root: str,
        class_list: List[str],
    ):
        self.root = root
        self.class_list = class_list
        self.class_to_idx = {name: i for i, name in enumerate(class_list)}

        self.samples: List[Tuple[str, int]] = []
        for class_name in os.listdir(root):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            idx = self.class_to_idx.get(class_name)
            if idx is None:
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Dict]:
        img_path, class_id = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        gt = {"class_id": class_id, "class_name": self.class_list[class_id],
              "image_path": img_path}
        return image, gt

    def get_gt_json(self) -> str:
        return ""
