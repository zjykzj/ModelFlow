# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : postprocess.py
@Author  : zj
@Description: 检测后处理 — NMS + 坐标缩放 + clip

支持 YOLOv5/v8/v11 输出格式差异：
- v5: (1, num_dets, 5+nc) - anchor-based
- v8/v11: (1, 84, 8400) - anchor-free（transposed）
"""

from typing import List, Optional

import numpy as np

from modelflow.interfaces import BasePostprocessor
from .ops import xywh2xyxy, nms, scale_boxes, clip_boxes


class DetectPostprocessor(BasePostprocessor):
    """检测后处理（NumPy 实现）

    Args:
        model_version: YOLO 版本（"v5", "v8", "v11"），决定输出解析方式
        class_list: 类别名称列表
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值
        max_det: 最大检测数
        input_shape: 网络输入 shape (H, W)
        original_shape: 原始图像 shape (H, W)
    """

    def __init__(
        self,
        model_version: str = "v8",
        class_list: Optional[List[str]] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        input_shape=(640, 640),
        original_shape=None,
    ):
        if model_version not in ("v5", "v8", "v11"):
            raise ValueError(f"Unknown model_version {model_version!r}. Options: 'v5', 'v8', 'v11'")
        self.model_version = model_version
        self.class_list = class_list
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.input_shape = input_shape
        self.original_shape = original_shape

    def __call__(self, raw: List[np.ndarray], **kwargs) -> dict:
        if not raw:
            raise ValueError("Empty raw output list — expected at least 1 output tensor")
        pred = raw[0]  # 取第一个输出

        if self.model_version in ("v8", "v11"):
            if pred.ndim != 3:
                raise ValueError(
                    f"Expected v8/v11 output shape (1, C, N), got {pred.shape}"
                )
        # v5: expect ndim >= 2; shape is (1, N, C)

        # 运行时 original_shape 优先（Pipeline 自动传入）
        original_shape = kwargs.get("original_shape", self.original_shape)

        if self.model_version in ("v8", "v11"):
            # v8/v11: (1, 84, 8400) → (1, 8400, 84)
            pred = pred.transpose(0, 2, 1)
        # v5: (1, num_dets, 85) — 无需转置

        pred = pred[0]  # (N, 85) 或 (N, 84)

        nc = pred.shape[1] - 4  # 类别数

        # 分离
        boxes = pred[:, :4]
        scores = pred[:, 4:] if nc == 1 else pred[:, 4:]

        if scores.ndim == 2:
            max_scores = scores.max(axis=1)
            class_ids = scores.argmax(axis=1)
        else:
            max_scores = scores
            class_ids = np.zeros_like(scores, dtype=int)

        # 阈值过滤
        mask = max_scores >= self.conf_thres
        boxes, scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return {"boxes": np.empty((0, 4)), "scores": np.empty(0),
                    "class_ids": np.empty(0, dtype=int)}

        # xywh → xyxy
        boxes = xywh2xyxy(boxes)

        # NMS
        keep = nms(boxes, scores, self.iou_thres)
        boxes, scores, class_ids = boxes[keep][:self.max_det], \
                                   scores[keep][:self.max_det], \
                                   class_ids[keep][:self.max_det]

        # 坐标缩放回原图
        if original_shape is not None:
            boxes = scale_boxes(self.input_shape, boxes, original_shape)

        result = {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids.astype(int),
        }

        if self.class_list is not None:
            result["class_names"] = [
                self.class_list[cid] if cid < len(self.class_list) else str(cid)
                for cid in result["class_ids"]
            ]

        return result
