# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : __init__.py
@Author  : zj
@Description: data 模块 — 数据集加载与配置

用法:
    from data import BaseDataset, build_dataset, get_class_names

    # 工厂函数
    ds = build_dataset("coco", path="/data/coco")

    # 直接使用
    from data import ClassifyDataset, COCODataset
"""

from data.base import BaseDataset
from data.classify import ClassifyDataset
from data.coco import COCODataset
from data.build import build_dataset, get_class_names, load_config

__all__ = [
    "BaseDataset",
    "ClassifyDataset",
    "COCODataset",
    "build_dataset",
    "get_class_names",
    "load_config",
]
