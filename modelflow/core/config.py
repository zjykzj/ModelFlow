# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : config.py
@Author  : zj
@Description: 配置管理

提供 Dataclass 方式的配置类，支持从字典和 YAML 加载。
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class ModelConfig:
    """模型推理配置"""
    model_path: str
    class_list: List[str]
    label_list: Optional[List[str]] = None
    backend: str = "onnxruntime"     # onnxruntime / tensorrt / triton
    processor: str = "numpy"          # numpy / torch
    task_type: str = "detect"         # classify / detect / segment / ...
    half: bool = False
    device: Optional[str] = None
    input_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**d)

    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        with open(json_path, "r") as f:
            return cls.from_dict(json.load(f))
