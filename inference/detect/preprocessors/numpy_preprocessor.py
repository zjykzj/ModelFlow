# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:10
@File    : numpy_preprocessor.py
@Author  : zj
@Description: 
"""

import cv2
import numpy as np

# Define constants for better readability and maintenance
DEFAULT_PAD_COLOR = (114, 114, 114)


def letterbox(im, new_shape=(640, 640), color=DEFAULT_PAD_COLOR, auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints.

    :param im: Input image in BGR format.
    :param new_shape: Target shape after padding and resizing.
    :param color: Padding color.
    :param auto: Minimum rectangle or not.
    :param scaleFill: Stretch to fill target shape or not.
    :param scaleup: Allow scaling up the image or only down.
    :param stride: Stride for making dimensions multiple of this value.
    :return: Resized and padded image, ratio, and padding values.
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    try:
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {e}")

    return im, ratio, (dw, dh)


class YOLOv5NumpyPreprocessor:
    def __init__(self, img_size=640, stride=32, auto=False):
        """
        Initialize the preprocessor with specific parameters.

        :param img_size: Target size for the shorter side of the image.
        :param stride: Stride for making dimensions multiple of this value.
        :param auto: Whether to use automatic padding calculation.
        """
        self.img_size = img_size
        self.stride = stride
        self.auto = auto

    def __call__(self, im0):
        """
        Process an input image for YOLOv5 model inference.

        :param im0: Original image in BGR format.
        :return: Preprocessed image ready for model input.
        """
        im, _, _ = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = im.astype(float)  # uint8 to fp16/32
        im /= 255.0  # Normalize to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im.astype(np.float32)
