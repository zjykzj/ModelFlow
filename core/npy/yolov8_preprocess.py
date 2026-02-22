# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 14:31
@File    : yolov8_preprocess.py
@Author  : zj
@Description: YOLOv8 图像检测预处理，支持 letterbox 保持长宽比

import cv2
from core.npy.yolov8_preprocess import ImgPrepare

# 初始化预处理器
preprocess = ImgPrepare(input_size=640, half=False)

# 加载图像 (BGR)
img = cv2.imread("test.jpg")
print(f"Original shape: {img.shape}")  # (H, W, 3)

# 预处理
im, ratio, pad = preprocess(img)
print(f"Preprocessed shape: {im.shape}")    # (1, 3, 640, 640)
print(f"Preprocessed dtype: {im.dtype}")    # float32
print(f"Ratio: {ratio}")                    # [[r, r]]
print(f"Padding: {pad}")                    # (dw, dh)

# FP16 模式
preprocess_half = ImgPrepare(input_size=640, half=True)
im_half, _, _ = preprocess_half(img)
print(f"FP16 dtype: {im_half.dtype}")       # float16

"""

import cv2
import numpy as np


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints (YOLOv8 标准 letterbox)

    Args:
        im: cv2 img (H, W, C), BGR
        new_shape: int or tuple(h, w), 目标尺寸
        color: fill color, 填充颜色 (B, G, R)
        auto: minimum rectangle, 是否使用最小矩形
        scaleFill: stretch image to new_shape, 是否拉伸填充
        scaleup: allow scaling up, 是否允许放大
        stride: model stride, 模型步长

    Returns:
        im: cv2 img, letterbox 后的图像
        r: Scale ratio (new / old), 缩放比例
        dw, dh: wh padding, 填充尺寸
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    if scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, r, dw, dh


def preprocess(image, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
               stride=32):
    """
    YOLOv8 图片预处理，与 Ultralytics 训练保持一致

    Args:
        image: cv2 img (H, W, C), BGR
        new_shape: 模型输入尺寸 int or tuple(h, w)
        color: 填充像素值 (B, G, R)
        auto: 是否使用最小矩形
        scaleFill: 是否拉伸填充
        scaleup: 是否允许放大
        stride: 模型步长

    Returns:
        im: np 数组, (1, 3, h, w) 模型输入数据, RGB, 归一化到 [0, 1]
        ratio: 缩放比例 (w_ratio, h_ratio)
        dw, dh: wh padding
    """
    # Letterbox 保持长宽比
    image, r, dw, dh = letterbox(image, new_shape, color=color, auto=auto, scaleFill=scaleFill, scaleup=scaleup,
                                 stride=stride)

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # HWC to CHW
    image = image.transpose((2, 0, 1))

    # Add batch dimension
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    # Normalize to [0, 1]
    im = image.astype(np.float32)
    im /= 255.0

    # YOLOv8 不需要 mean/std 归一化，只需 /255
    return im, (r, r), (dw, dh)


class ImgPrepare:
    """
    YOLOv8 图像预处理类，统一接口

    Examples:
        >>> from core.npy.yolov8_preprocess import ImgPrepare
        >>> import cv2
        >>>
        >>> preprocess = ImgPrepare(input_size=640, half=False)
        >>> img = cv2.imread("test.jpg")
        >>> im, ratio, padding, im_shape, im0_shape = preprocess(img)
        >>> print(im.shape)  # (1, 3, 640, 640)
        >>> print(im_shape)  # (640, 640)
        >>> print(im0_shape)  # (480, 640)
    """

    def __init__(self, input_size=640, half=False, auto=False, stride=32):
        """
        Args:
            input_size: int, 模型输入尺寸（正方形）
            half: bool, 是否使用 FP16
            auto: bool, 是否使用最小矩形 padding
            stride: int, 模型步长
        """
        self.input_size = input_size
        self.half = half
        self.auto = auto
        self.stride = stride

    def __call__(self, image):
        """
        Args:
            image: cv2 img (H, W, C), BGR

        Returns:
            im: np 数组, (1, 3, h, w), dtype float32/float16
            ratio: np 数组, (1, 2), (w_ratio, h_ratio)
            padding: tuple, (dw, dh)
            im_shape: tuple, (h, w), 模型输入形状
            im0_shape: tuple, (h, w), 原始图像形状
        """
        # 记录原始图像形状
        im0_shape = image.shape[:2]  # (h, w)

        # 预处理
        im, ratio, pad = preprocess(
            image,
            new_shape=(self.input_size, self.input_size),
            auto=self.auto,
            stride=self.stride
        )

        # 获取模型输入形状
        im_shape = im.shape[2:]  # (h, w)

        # 可选：FP16
        if self.half:
            im = im.astype(np.float16)

        return im, np.array([ratio], dtype=np.float32), pad, im_shape, im0_shape
