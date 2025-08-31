# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/29 16:48
@File    : general.py
@Author  : zj
@Description: 
"""

LOGGING_NAME = "yolov5"

MODEL_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
               22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
               29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
               35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
               40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
               48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
               55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
               62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
               69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
               76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

CLASSES_NAME = [item[1] for item in MODEL_NAMES.items()]

# --------------------------------------------------------------------------------------- Draw


import cv2
import random
import colorsys


def gen_colors(classes):
    """
        generate unique hues for each class and convert to bgr
        classes -- list -- class names (80 for coco dataset)
        -> list
    """
    hsvs = []
    for x in range(len(classes)):
        hsvs.append([float(x) / len(classes), 1., 0.7])
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = []
    for hsv in hsvs:
        h, s, v = hsv
        rgb = colorsys.hsv_to_rgb(h, s, v)
        rgbs.append(rgb)
    bgrs = []
    for rgb in rgbs:
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        bgrs.append(bgr)
    return bgrs


# def draw_results(img, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True):
#     CLASSES_COLOR = gen_colors(CLASSES_NAME)
#
#     overlay = img.copy()
#     if len(boxes) != 0:
#         for box, conf, cls in zip(boxes, confs, cls_ids):
#             if is_xyxy:
#                 x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#             else:
#                 # xywh
#                 x1, y1, box_w, box_h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#                 x2 = x1 + box_w
#                 y2 = y1 + box_h
#             cls_name = CLASSES_NAME[int(cls)]
#             color = CLASSES_COLOR[int(cls)]
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
#             cv2.putText(overlay, '%s %.3f' % (cls_name, conf), org=(x1, int(y1 - 10)),
#                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                         fontScale=1, color=color)
#     return overlay

import cv2
import numpy as np
import torch


def draw_results(img, boxes, confs, cls_ids, masks=None, CLASSES_NAME=None, CLASSES_COLOR=None, is_xyxy=True,
                 alpha=0.5):
    """
    在图像上绘制检测框和可选的分割掩码。

    参数:
    - img: numpy array (H, W, 3), BGR 格式图像
    - boxes: list 或 numpy array, 形状为 [N, 4], 检测框坐标
    - confs: list 或 numpy array, 形状为 [N], 置信度
    - cls_ids: list 或 numpy array, 形状为 [N], 类别索引
    - masks: torch.Tensor 或 None, 形状为 [N, H, W] 或 [N, mask_h, mask_w]，0/1 二值 mask
             如果 mask 尺寸不是原图大小，会自动缩放到图像大小
    - CLASSES_NAME: list of str, 类别名称列表
    - CLASSES_COLOR: list of tuple, 每个类别的颜色 (B, G, R)，可选，若为 None 则自动生成
    - is_xyxy: bool, 框是否为 (x1, y1, x2, y2) 格式；否则为 (x, y, w, h)
    - alpha: float, 分割 mask 的透明度权重

    返回:
    - overlayed_img: 绘制后的图像 (numpy array)
    """
    if CLASSES_NAME is None:
        CLASSES_NAME = ['class_' + str(i) for i in range(1000)]

    if CLASSES_COLOR is None:
        CLASSES_COLOR = gen_colors(CLASSES_NAME)  # 假你有这个函数生成颜色

    # 确保图像是可修改的副本
    img = img.copy()
    overlay = img.copy()
    h, w = img.shape[:2]

    if masks is not None and len(masks) > 0:
        masks = masks.float()
        if masks.dim() == 3:
            if masks.shape[-2] != h or masks.shape[-1] != w:
                print(
                    "asdfasdfasdfasdf"
                )
                masks = masks.unsqueeze(0)
                masks = torch.nn.functional.interpolate(
                    masks, size=(h, w), mode='bilinear', align_corners=False
                )
                masks = (masks > 0.5).squeeze(0)
            for i, mask in enumerate(masks):
                color_idx = cls_ids[i] if len(cls_ids) > i else 0
                color = CLASSES_COLOR[int(color_idx)]

                # 增加边框以突出显示
                contours, _ = cv2.findContours(mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)

                mask_np = mask.cpu().numpy().astype(np.uint8)
                color_mask = np.zeros_like(overlay, dtype=np.uint8)
                color_mask[:] = color

                masked_region = cv2.bitwise_and(color_mask, color_mask, mask=mask_np)
                cv2.addWeighted(overlay, 1.0, masked_region, alpha, 0, overlay)

        # 如果需要进一步调整边界框的绘制逻辑，请确保坐标转换是正确的
    if len(boxes) > 0:
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, cls_ids)):
            if is_xyxy:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            else:
                x, y, w_box, h_box = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x1, y1, x2, y2 = x, y, x + w_box, y + h_box

            # 确保坐标在图像范围内
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            cls_name = CLASSES_NAME[int(cls)] if CLASSES_NAME else str(int(cls))
            color = CLASSES_COLOR[int(cls)]
            # 绘制边界框
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

            # 绘制标签文本
            label = f'{cls_name} {float(conf):.2f}'
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2)
            cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(overlay, label, org=(x1, y1 - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 合并最终图像
    result = cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)

    return result
