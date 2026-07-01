# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : postprocess.py
@Author  : zj
@Description: 实例分割后处理 — NMS + process_mask + crop_mask + scale_image

实例分割模型的输出包含：
- pred: (1, 116, 8400) — 前 84 维同检测，后 32 维为 mask 系数
- proto: (1, 32, 160, 160) — prototype mask
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from modelflow.interfaces import BasePostprocessor
from modelflow.processors.detect.ops import xywh2xyxy, nms, scale_boxes, clip_boxes


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """根据 boxes 裁剪 masks"""
    n, h, w = masks.shape
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    rows = np.arange(h)[None, :, None]  # (1, h, 1)
    cols = np.arange(w)[None, None, :]  # (1, 1, w)
    mask = (rows >= y1[:, None, None]) & (rows <= y2[:, None, None]) & \
           (cols >= x1[:, None, None]) & (cols <= x2[:, None, None])
    return masks * mask


def process_mask(protos: np.ndarray, mask_coeffs: np.ndarray, bboxes: np.ndarray,
                 shape: Tuple[int, int], upsample: bool = False) -> np.ndarray:
    """从 prototype 和 mask 系数生成最终 mask

    Args:
        protos: (1, 32, 160, 160)
        mask_coeffs: (N, 32)
        bboxes: (N, 4)
        shape: (H, W) 目标尺寸
        upsample: 是否上采样到 shape

    Returns:
        masks: (N, H, W) uint8 掩码
    """
    protos = protos[0]  # (32, 160, 160)
    masks = np.tensordot(mask_coeffs, protos, axes=1)  # (N, 160, 160)
    masks = 1.0 / (1.0 + np.exp(-masks))  # sigmoid

    if upsample:
        masks_resized = []
        for m in masks:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            masks_resized.append(m)
        masks = np.array(masks_resized)
    else:
        masks = np.array([cv2.resize(m, (shape[1], shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                          for m in masks])

    masks = (masks > 0.5).astype(np.uint8)
    masks = crop_mask(masks, bboxes)
    return masks


class SegmentPostprocessor(BasePostprocessor):
    """实例分割后处理（NumPy 实现）

    Args:
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值
        max_det: 最大检测数
        class_list: 类别名称列表
        input_shape: 网络输入 shape (H, W)
        original_shape: 原始图像 shape (H, W)
    """

    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        class_list: Optional[List[str]] = None,
        input_shape=(640, 640),
        original_shape=None,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.class_list = class_list
        self.input_shape = input_shape
        self.original_shape = original_shape

    def __call__(self, raw: List[np.ndarray], **kwargs) -> dict:
        pred = raw[0]  # (1, 116, 8400)
        proto = raw[1] if len(raw) > 1 else None  # (1, 32, 160, 160)

        # 运行时 original_shape 优先
        original_shape = kwargs.get("original_shape", self.original_shape)

        pred = pred.transpose(0, 2, 1)[0]  # (8400, 116)

        nc = pred.shape[1] - 4 - 32  # = 80 for COCO

        boxes = pred[:, :4]
        scores = pred[:, 4:4 + nc]
        mask_coeffs = pred[:, 4 + nc:]  # (N, 32)

        max_scores = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)

        # 阈值过滤
        mask = max_scores >= self.conf_thres
        boxes, scores, class_ids, mask_coeffs = \
            boxes[mask], max_scores[mask], class_ids[mask], mask_coeffs[mask]

        if len(boxes) == 0:
            return {"boxes": np.empty((0, 4)), "scores": np.empty(0),
                    "class_ids": np.empty(0, dtype=int), "masks": None}

        boxes = xywh2xyxy(boxes)

        keep = nms(boxes, scores, self.iou_thres)
        boxes, scores, class_ids, mask_coeffs = \
            boxes[keep][:self.max_det], scores[keep][:self.max_det], \
            class_ids[keep][:self.max_det], mask_coeffs[keep][:self.max_det]

        # 缩放 boxes
        orig_shape = original_shape or self.input_shape
        boxes = scale_boxes(self.input_shape, boxes, orig_shape)

        result = {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids.astype(int),
            "masks": None,
        }

        if self.class_list is not None:
            result["class_names"] = [
                self.class_list[cid] if cid < len(self.class_list) else str(cid)
                for cid in result["class_ids"]
            ]

        # 生成 masks
        if proto is not None and len(boxes) > 0:
            result["masks"] = process_mask(
                proto, mask_coeffs, boxes, orig_shape, upsample=True
            )

        return result
