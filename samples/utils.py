# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : utils.py
@Author  : zj
@Description: 可视化工具 — OpenCV 绘制检测框 / 分割 mask / 分类标签

纯 OpenCV 实现，无额外依赖（opencv-python 已是 modelflow core dep）。
"""

from typing import List, Optional

import cv2
import numpy as np

# 固定调色板（BGR 格式，高对比度 20 色）
_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128),
    (0, 128, 128), (128, 0, 128), (128, 128, 0), (64, 64, 255),
    (64, 255, 64), (255, 64, 64), (64, 255, 255), (255, 255, 64),
]


def _get_color(idx: int) -> tuple:
    return _PALETTE[idx % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_list: Optional[List[str]] = None,
    **kwargs,
) -> np.ndarray:
    """在 BGR 图像上绘制检测框。

    Args:
        image: BGR 图像 (H, W, 3), uint8
        boxes: 边界框数组 (N, 4), xyxy 格式
        scores: 置信度数组 (N,)
        class_ids: 类别 ID 数组 (N,)
        class_list: 类别名称列表（可选）
        **kwargs: thickness (default 2), font_scale (default 0.5)

    Returns:
        绘制后的 BGR 图像副本
    """
    canvas = image.copy()
    thickness = kwargs.get("thickness", 2)
    font_scale = kwargs.get("font_scale", 0.5)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        score = scores[i] if i < len(scores) else 1.0
        cls_id = int(class_ids[i]) if i < len(class_ids) else 0

        color = _get_color(cls_id)

        # 框
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # 标签
        label_parts = []
        if class_list and cls_id < len(class_list):
            label_parts.append(class_list[cls_id])
        else:
            label_parts.append(str(cls_id))
        label_parts.append(f"{score:.2f}")
        label = " ".join(label_parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(y1 - th - 4, th + 4)
        cv2.rectangle(canvas, (x1, label_y - th - 2), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(canvas, label, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return canvas


def draw_segmentation(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_list: Optional[List[str]] = None,
    **kwargs,
) -> np.ndarray:
    """在 BGR 图像上绘制分割结果（mask overlay + bbox）。

    Args:
        image: BGR 图像 (H, W, 3), uint8
        boxes: 边界框数组 (N, 4), xyxy 格式
        masks: mask 数组 (N, H, W), bool 或 float
        scores: 置信度数组 (N,)
        class_ids: 类别 ID 数组 (N,)
        class_list: 类别名称列表（可选）
        **kwargs: alpha (default 0.4), thickness, font_scale

    Returns:
        绘制后的 BGR 图像副本
    """
    canvas = image.copy()
    alpha = kwargs.get("alpha", 0.4)
    overlay = canvas.copy()

    for i in range(len(masks)):
        cls_id = int(class_ids[i]) if i < len(class_ids) else 0
        color = _get_color(cls_id)
        mask = masks[i].astype(bool)
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

    # 画框 + 标签（复用 detection 绘制逻辑）
    return draw_detections(canvas, boxes, scores, class_ids, class_list, **kwargs)


def draw_classification(
    image: np.ndarray,
    class_names: List[str],
    scores: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """在 BGR 图像上叠加分类结果。

    Args:
        image: BGR 图像 (H, W, 3), uint8
        class_names: 类别名称列表（top-k）
        scores: 对应的置信度数组
        **kwargs: font_scale (default 0.6), max_lines (default 5)

    Returns:
        绘制后的 BGR 图像副本
    """
    canvas = image.copy()
    font_scale = kwargs.get("font_scale", 0.6)
    max_lines = kwargs.get("max_lines", 5)
    thickness = 1

    # 背景面板
    panel_h = int(28 * min(max_lines, len(class_names)) + 16)
    panel_w = 280
    cv2.rectangle(canvas, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    alpha = 0.6
    canvas[8:8 + panel_h, 8:8 + panel_w] = (
        canvas[8:8 + panel_h, 8:8 + panel_w] * (1 - alpha) + np.array([0, 0, 0]) * alpha
    ).astype(np.uint8)

    # 文本
    y = 32
    for i in range(min(max_lines, len(class_names))):
        label = f"{class_names[i]}: {scores[i]:.4f}"
        cv2.putText(canvas, label, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y += 28

    return canvas
