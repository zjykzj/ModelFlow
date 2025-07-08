# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:10
@File    : torch_postprocessor.py
@Author  : zj
@Description: 
"""

import torch


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box = xywh2xyxy(x[:, :4])
        conf = x[:, 4:5]
        cls = x[:, 5:]
        x = torch.cat((box, conf, cls), 1)

        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]

    return output


def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


class YOLOv5TorchPostprocessor:
    def __init__(self, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, outputs, original_shape):
        dets = non_max_suppression(outputs, self.conf_thres, self.iou_thres)
        results = []
        for det in dets:
            result = {
                "boxes": det[:, :4].cpu().numpy(),
                "scores": det[:, 4].cpu().numpy(),
                "class_ids": det[:, 5].cpu().int().numpy()
            }
            results.append(result)
        return results
