# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 10:19
@File    : preprocess_classify.py
@Author  : zj
@Description:

from PIL import Image
from preprocess_classify import ImgPrepare

img = Image.open("cat.jpg").convert("RGB")

# ==================== 模式1: 直接缩放 ====================
preprocess_resize = ImgPrepare(input_size=224, batch=True, mode="resize")
input1 = preprocess_resize(img)
print(f"Resize mode output shape: {input1.shape}")  # (1, 3, 224, 224)

# ==================== 模式2: 缩放+中央裁剪（ImageNet 标准）====================
preprocess_crop = ImgPrepare(input_size=256, crop_size=224, batch=True, mode="crop")
input2 = preprocess_crop(img)
print(f"Crop mode output shape: {input2.shape}")  # (1, 3, 224, 224)

"""

import numpy as np
from PIL import Image


class ImgPrepare:

    def __init__(self, input_size=224, crop_size=None, half=False, batch=False, mode="resize"):
        """
        Args:
            input_size: int, 目标尺寸
            crop_size: int, 裁剪尺寸（用于 crop 模式），默认等于 input_size
            half: bool, 是否使用 FP16
            batch: bool, 是否添加 batch 维度
            mode: str, 预处理模式
                - "resize": 直接缩放到 input_size×input_size
                - "crop": 先缩放短边到 input_size，再中央裁剪到 crop_size×crop_size（ImageNet 标准）
        """
        self.input_size = input_size
        self.crop_size = crop_size if crop_size is not None else input_size
        self.half = half
        self.batch = batch
        self.mode = mode

        # ImageNet 标准化参数
        self.mean = np.array( [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, image):
        """
        Args:
            image: 可以是 PIL.Image.Image 或 numpy.ndarray (H, W, C), uint8, RGB

        Returns:
            np.ndarray: shape (C, H, W) 或 (1, C, H, W), dtype float32, normalized with ImageNet stats
        """
        # 1. 确保输入是 PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be PIL.Image or numpy.ndarray (RGB)")

        # 2. 根据模式进行几何变换
        if self.mode == "resize":
            # 模式1: 直接缩放
            resized = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        elif self.mode == "crop":
            # 模式2: 先缩放短边到 input_size，再中央裁剪（ImageNet 标准）
            resized = self._resize_short_edge(image, self.input_size)
            resized = self._center_crop(resized, self.crop_size)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'resize' or 'crop'")

        # 3. Convert to numpy array in [0, 255] -> then to [0.0, 1.0]
        img_np = np.array(resized).astype(np.float32) / 255.0  # Shape: (H, W, C)

        # 4. Normalize with ImageNet mean and std (applied per channel)
        img_normalized = (img_np - self.mean) / self.std  # Broadcasting over H, W

        # 5. Convert to (C, H, W) like PyTorch tensor
        img_chw = np.transpose(img_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        # 6. 可选：FP16 和 batch 维度
        if self.half:
            img_chw = img_chw.astype(np.float16)

        if self.batch:
            img_chw = img_chw[None]  # (C, H, W) -> (1, C, H, W)

        return img_chw

    def _resize_short_edge(self, image, size):
        """
        缩放图像，使短边等于 size，长边等比例缩放
        等效于 torchvision.transforms.Resize(size)
        """
        width, height = image.size
        if width <= height:
            # 宽是短边
            new_width = size
            new_height = int(size * height / width)
        else:
            # 高是短边
            new_height = size
            new_width = int(size * width / height)

        return image.resize((new_width, new_height), resample=Image.BILINEAR)

    def _center_crop(self, image, size):
        """
        中央裁剪到 size×size
        等效于 torchvision.transforms.CenterCrop(size)
        """
        width, height = image.size
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size

        return image.crop((left, top, right, bottom))
