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


def draw_results(img, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True):
    CLASSES_COLOR = gen_colors(CLASSES_NAME)

    overlay = img.copy()
    if len(boxes) != 0:
        for box, conf, cls in zip(boxes, confs, cls_ids):
            if is_xyxy:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            else:
                # xywh
                x1, y1, box_w, box_h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x2 = x1 + box_w
                y2 = y1 + box_h
            cls_name = CLASSES_NAME[int(cls)]
            color = CLASSES_COLOR[int(cls)]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(overlay, '%s %.3f' % (cls_name, conf), org=(x1, int(y1 - 10)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=color)
    return overlay
