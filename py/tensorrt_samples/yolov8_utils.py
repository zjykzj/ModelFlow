# -*- coding: utf-8 -*-

"""
@Time    : 2023/10/30 20:13
@File    : yolov8_utils.py
@Author  : zj
@Description: 
"""

import numpy as np


def rescale_boxes(boxes, input_shape, image_shape):
    # Rescale boxes to original image dimensions
    input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
    return boxes


def xywh2xyxyV8(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def extract_boxes(box_predictions, input_h, input_w, img_h, img_w):
    boxes = box_predictions[:, :4]
    boxes = rescale_boxes(boxes, (input_h, input_w), (img_h, img_w))
    boxes = xywh2xyxyV8(boxes)
    # Check the boxes are within the image
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)
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


def det_process_box_output(box_output, conf_threshold, iou_threshold, input_h, input_w, img_h, img_w):
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
    boxes = extract_boxes(box_predictions, input_h, input_w, img_h, img_w)
    # Apply nms filtering
    indices = nmsV8(boxes, scores, iou_threshold)
    return boxes[indices], scores[indices], class_ids[indices]
