# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : config.py
@Author  : zj
@Description: 模型推理配置 & 类别标签定义
"""

from dataclasses import dataclass
from typing import List, Optional


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


# ---------------------------------------------------------------------------
# COCO 80 类（detect / segment 任务使用）
# ---------------------------------------------------------------------------

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

COCO_NUM_CLASSES = len(COCO_CLASSES)  # 80


# ---------------------------------------------------------------------------
# ImageNet 1000 类
# ---------------------------------------------------------------------------

# Placeholder（实际使用时通过 torchvision 加载完整映射）
_IMAGENET_PLACEHOLDER = ["n%08d" % i for i in range(1000)]


def get_imagenet_classes() -> list:
    """获取 ImageNet 1000 类名称列表

    生产环境下建议从文件加载完整映射：
        from torchvision.models import ResNet18_Weights
        categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    """
    try:
        from torchvision.models import ResNet18_Weights
        categories = ResNet18_Weights.IMAGENET1K_V1.meta.get("categories")
        if categories:
            return categories
    except (ImportError, AttributeError):
        pass
    return _IMAGENET_PLACEHOLDER
