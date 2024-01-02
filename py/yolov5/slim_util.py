# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/27 10:51
@File    : thin_util.py
@Author  : zj
@Description: 
"""

import time

import torch
import torchvision

from general import LOGGER
from torch_util import xywh2xyxy, box_iou


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    # prediction的shape大小为[Batch, Det_num, xywh+conf+Num_classes]

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    # 置信度过滤，得到候选列表tensor([[False, False, False,  ..., False, False, False]])
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    # 设置最大的预测框面积
    max_wh = 7680  # (pixels) maximum box width and height
    # 执行NMS之前，保留最大的预测框数目
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    # 时间限制
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    # 冗余检测？
    redundant = True  # require redundant detections
    # 多标签?
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # NMS类别
    merge = False  # use merge-NMS

    t = time.time()
    # 这个掩码不知道干嘛的？
    mi = 5 + nc  # mask start index
    # 指定输出格式，这样保证
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # 获取第xi张图片，经过置信度阈值过滤的结果

        # 设置了标签，估计是标签合并，先不管
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # 经过目标置信度过滤之后，没有符合条件的预测框了，跳过预测下一条
        # If none remain process next image
        if not x.shape[0]:
            continue

        # 置信度计算 = 目标置信度 * 分类置信度
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # xywh -> xyxy
        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            # 选择每个预测框最大的置信度和对应类别下标
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            # 再次进行置信度过滤，合并xyxy + conf + cls_id
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # 判断剩余预测框数目，如果大于最大设置，根据置信度大小截断
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # 区分不同类别的边界框坐标，分类别进行NMS?还是统一进行NMS?
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # 如果经过NMS之后的结果仍旧超过最大约束，进行截断
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        # 赋值第xi个图片的后处理结果到output
        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def non_max_suppression_v1(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    最基本流程：

    1. 确保输入prediction格式为[batch, pred_num, 85(xywh+obj_conf+pred_cls)]
    2. 获取类别数/批量大小
    3. 目标置信度过滤
    4. 逐个图像执行
        4.1. 获取目标置信度过滤结果
        4.2. 如果没有符合条件的预测框，跳过
        4.3. xywh -> xyxy
        4.4. 获取置信度（目标置信度*分类概率）和最大分类概率的类别下标
        4.5. 合并每个预测框的xyxy、置信度和类别下标，在此进行置信度过滤
        4.6. 区分不同类别的边界框坐标，执行NMS
        4.7. 赋值过滤最后的预测框到output[xi]

    1. 确保输入prediction格式为[batch, pred_num, 85(xywh+obj_conf+pred_cls)]
    3. 目标置信度过滤
    4. 逐个图像执行
        4.1. 获取目标置信度过滤结果
        4.2. 如果没有符合条件的预测框，跳过

        第一步 计算置信度，进行置信度过滤
        第二步 逐类别执行NMS IOU过滤，生成候选下标列表
        第三步：赋值经过目标置信度过滤、置信度过滤和IOU过滤后最终的预测框结果。

        4.3. xywh -> xyxy
        4.4. 获取置信度（目标置信度*分类概率）和最大分类概率的类别下标
        4.5. 合并每个预测框的xyxy、置信度和类别下标，在此进行置信度过滤
        4.6. 区分不同类别的边界框坐标，执行NMS（逐个类别执行NMS）
        4.7. 赋值过滤最后的预测框到output[xi]

    优化一：实施max_nms截断
    优化二：

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    assert len(prediction.shape) == 3, prediction.shape
    # prediction的shape大小为[Batch, Det_num, xywh+conf+Num_classes]

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    # 置信度过滤，得到候选列表tensor([[False, False, False,  ..., False, False, False]])
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    # 设置最大的预测框面积
    max_wh = 7680  # (pixels) maximum box width and height
    # 执行NMS之前，保留最大的预测框数目
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    # 时间限制
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    # 冗余检测？
    redundant = True  # require redundant detections
    # 多标签?
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # NMS类别
    merge = False  # use merge-NMS

    t = time.time()
    # 这个掩码不知道干嘛的？
    mi = 5 + nc  # mask start index
    # 指定输出格式，这样保证
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # 获取第xi张图片，经过置信度阈值过滤的结果

        # 设置了标签，估计是标签合并，先不管
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # 经过目标置信度过滤之后，没有符合条件的预测框了，跳过预测下一条
        # If none remain process next image
        if not x.shape[0]:
            continue

        # 置信度计算 = 目标置信度 * 分类置信度
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # xywh -> xyxy
        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            # 选择每个预测框最大的置信度和对应类别下标
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            # 再次进行置信度过滤，合并xyxy + conf + cls_id
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # 判断剩余预测框数目，如果大于最大设置，根据置信度大小截断
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # 区分不同类别的边界框坐标，分类别进行NMS?还是统一进行NMS?
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # 如果经过NMS之后的结果仍旧超过最大约束，进行截断
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        # 赋值第xi个图片的后处理结果到output
        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output
