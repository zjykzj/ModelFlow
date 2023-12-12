# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/12 16:10
@File    : numpy_util.py
@Author  : zj
@Description: 
"""

import numpy as np


def xywh2xyxyV8(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def extract_boxes(box_predictions, input_h, input_w):
    boxes = box_predictions[:, :4]
    boxes = xywh2xyxyV8(boxes)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, input_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, input_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, input_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, input_h)
    return boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    # Compute IoU
    iou = intersection_area / union_area
    return iou


def nmsV8(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]
        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes


def det_process_box_output(box_output, conf_threshold, iou_threshold, input_h, input_w):
    predictions = np.squeeze(box_output).T
    num_classes = box_output.shape[1] - 4  # proto mask (1, 32, 160, 160) + 4
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:4 + num_classes], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]
    if len(scores) == 0:
        return [], [], []
    box_predictions = predictions[..., :num_classes + 4]
    # Get the class with the highest confidence
    class_ids = np.argmax(box_predictions[:, 4:], axis=1)
    # Get bounding boxes for each object
    boxes = extract_boxes(box_predictions, input_h, input_w)

    # Apply nms filtering
    indices = nmsV8(boxes, scores, iou_threshold)
    return boxes[indices], scores[indices], class_ids[indices]


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    # if isinstance(boxes, torch.Tensor):  # faster individually
    #     boxes[..., 0].clamp_(0, shape[1])  # x1
    #     boxes[..., 1].clamp_(0, shape[0])  # y1
    #     boxes[..., 2].clamp_(0, shape[1])  # x2
    #     boxes[..., 3].clamp_(0, shape[0])  # y2
    # else:  # np.array (faster grouped)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
