# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : build.py
@Author  : zj
@Description: Dataset 工厂函数 — 从 YAML 配置构建数据集实例

用法:
    from data import build_dataset, get_class_names

    # 按名称加载
    ds = build_dataset("coco", path="/data/coco")
    class_names = get_class_names("coco")

    # 从指定 YAML 文件加载
    ds = build_dataset("/path/to/custom.yaml")
"""

import os
from typing import Dict, List, Optional

import yaml

from data.base import BaseDataset
from data.classify import ClassifyDataset
from data.coco import COCODataset

# 内置配置目录
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _resolve_config(config: str) -> str:
    """解析配置名或路径 → YAML 文件路径

    Args:
        config: 配置名（如 "coco"）或 YAML 文件路径

    Returns:
        YAML 文件的绝对路径
    """
    if config.endswith((".yaml", ".yml")) and os.path.isfile(config):
        return config

    # 尝试从内置 configs/ 目录加载
    for ext in (".yaml", ".yml"):
        path = os.path.join(_CONFIG_DIR, config + ext)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Dataset config '{config}' not found. "
        f"Available: {_list_builtin_configs()}"
    )


def _list_builtin_configs() -> List[str]:
    """列出内置配置名"""
    if not os.path.isdir(_CONFIG_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_CONFIG_DIR)
        if f.endswith((".yaml", ".yml"))
    )


def _load_yaml(yaml_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config: str) -> dict:
    """加载并解析数据集配置（不构建实例）

    Args:
        config: 配置名或 YAML 路径

    Returns:
        解析后的配置 dict
    """
    path = _resolve_config(config)
    return _load_yaml(path)


def get_class_names(config: str) -> List[str]:
    """从 YAML 配置中提取类别名列表

    Args:
        config: 配置名或 YAML 路径

    Returns:
        类别名称列表（按类别 ID 索引）
    """
    cfg = load_config(config)
    names = cfg.get("names", {})
    # names 可能是 {0: "person", 1: "car"} 或 ["person", "car"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def build_dataset(
    config: str = "coco",
    task: Optional[str] = None,
    **overrides,
) -> BaseDataset:
    """从 YAML 配置构建数据集实例

    Args:
        config: 配置名（如 "coco", "imagenet"）或 YAML 文件路径
        task: 任务类型（"classify", "detect", "segment"），None 则读取 YAML 中的 task 字段
        **overrides: 覆盖 YAML 配置字段
            path: 数据集根目录
            val: 验证集子目录
            anno: 标注 JSON 相对路径
            等

    Returns:
        BaseDataset 实例

    Examples:
        # 按名称加载
        ds = build_dataset("coco", path="/data/coco")

        # 覆盖子目录
        ds = build_dataset("coco", path="/data/coco128", val="images/train2017")

        # 指定任务类型（覆盖 YAML）
        ds = build_dataset("coco-seg", task="segment")

        # 从自定义 YAML 加载
        ds = build_dataset("./my_dataset.yaml")
    """
    cfg = load_config(config)

    # 合并 overrides
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    task = task or cfg.get("task", "detect")
    class_list = get_class_names(config)

    # 如果在 overrides 中覆盖了 names，使用覆盖后的值
    if "names" in overrides:
        names = overrides["names"]
        class_list = list(names.values()) if isinstance(names, dict) else list(names)

    # 构建路径
    root_path = cfg.get("path", "")
    val_dir = os.path.join(root_path, cfg.get("val", ""))

    anno_json = None
    if "anno" in cfg:
        anno_json = os.path.join(root_path, cfg["anno"])

    if task == "classify":
        return ClassifyDataset(root=val_dir, class_list=class_list)
    elif task in ("detect", "segment"):
        return COCODataset(
            img_dir=val_dir,
            class_list=class_list,
            task=task,
            anno_json=anno_json,
        )
    else:
        raise ValueError(f"Unsupported task: {task}. Supported: classify, detect, segment")
