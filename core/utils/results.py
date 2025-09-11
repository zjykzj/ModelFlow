# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/11 19:50
@File    : results.py
@Author  : zj
@Description: 
"""

from pathlib import Path

from core.utils.ops import xyxy2xywh


def save_txt(boxes, confs, cls_ids, im0_shape, txt_file, save_conf=False, segments=None):
    texts = []
    for i, (xyxy, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
        xywh = xyxy2xywh(xyxy)
        xywh[..., [0, 2]] /= im0_shape[1]
        xywh[..., [1, 3]] /= im0_shape[0]

        line = (cls_id, *xywh)
        if segments is not None:
            line = (cls_id, *segments.reshape(-1))
        if save_conf:
            # [cls_id, x_c, y_c, box_w, box_h, conf]
            line += (conf,)
        texts.append(("%g " * len(line)).rstrip() % line)

    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, "w") as f:
            f.writelines(text + "\n" for text in texts)
