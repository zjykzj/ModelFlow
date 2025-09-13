# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/7 19:50
@File    : results.py
@Author  : zj
@Description: 
"""

import numpy as np
from pathlib import Path


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (numpy.ndarray): Clipped coordinates
    """
    # np.array (faster grouped)
    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def save_txt(boxes, confs, cls_ids, im0_shape, txt_file, save_conf=False, segments=None):
    texts = []
    for i, (xyxy, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
        xywh = xyxy2xywh(xyxy)
        xywh[..., [0, 2]] /= im0_shape[1]
        xywh[..., [1, 3]] /= im0_shape[0]

        line = (cls_id, *xywh)
        if segments is not None:
            segment = clip_coords(segments[i], im0_shape)
            segment[..., 0] /= im0_shape[1]  # width
            segment[..., 1] /= im0_shape[0]  # height

            line = (cls_id, *segment.reshape(-1))
        if save_conf:
            # [cls_id, x_c, y_c, box_w, box_h, conf]
            line += (conf,)
        texts.append(("%g " * len(line)).rstrip() % line)

    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, "w") as f:
            f.writelines(text + "\n" for text in texts)
