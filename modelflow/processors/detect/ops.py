# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : ops.py
@Author  : zj
@Description: 检测后处理共享算子

这些算子在 postprocessor 和 evaluator 中通用。
"""
import numpy as np


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """(x_center, y_center, w, h) → (x1, y1, x2, y2)"""
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """非极大值抑制（NumPy 实现）

    Args:
        boxes: (N, 4) xyxy 格式
        scores: (N,) 置信度
        iou_thresh: IoU 阈值

    Returns:
        保留的索引数组
    """
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]

        # 计算 IoU
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-7)

        order = rest[iou <= iou_thresh]

    return np.array(keep, dtype=int)


def scale_boxes(img1_shape, boxes, img0_shape):
    """将 boxes 从网络输入尺寸缩放到原图尺寸"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
           (img1_shape[0] - img0_shape[0] * gain) / 2)

    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain

    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes: np.ndarray, shape):
    """将 boxes 限制在图像边界内"""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
