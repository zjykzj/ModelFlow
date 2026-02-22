# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 10:19
@File    : preprocess_classify.py
@Author  : zj
@Description: PyTorch 版本图像分类预处理，支持 resize 和 crop 两种模式

from PIL import Image
from preprocess_classify import ImgPrepare

img = Image.open("cat.jpg").convert("RGB")

# ==================== 模式1: 直接缩放 ====================
preprocess_resize = ImgPrepare(input_size=224, batch=True, mode="resize")
input1 = preprocess_resize(img)
print(f"Resize mode output shape: {input1.shape}")  # torch.Size([1, 3, 224, 224])
print(f"Resize mode dtype: {input1.dtype}")  # torch.float32

# ==================== 模式2: 缩放+中央裁剪（ImageNet 标准）====================
preprocess_crop = ImgPrepare(input_size=256, crop_size=224, batch=True, mode="crop")
input2 = preprocess_crop(img)
print(f"Crop mode output shape: {input2.shape}")  # torch.Size([1, 3, 224, 224])
print(f"Crop mode dtype: {input2.dtype}")  # torch.float32

# ==================== FP16 模式 ====================
preprocess_half = ImgPrepare(input_size=224, batch=True, half=True, mode="resize")
input3 = preprocess_half(img)
print(f"FP16 mode dtype: {input3.dtype}")  # torch.float16

# ==================== 支持 numpy 输入 ====================
import numpy as np
img_np = np.array(img)  # (H, W, C), uint8
input4 = preprocess_resize(img_np)
print(f"Numpy input output shape: {input4.shape}")  # torch.Size([1, 3, 224, 224])

"""

import torch
import torchvision
from torchvision import transforms


class ImgPrepare:

    def __init__(self, input_size=224, crop_size=None, half=False, batch=False, mode="resize"):
        """
        Args:
            input_size: int, 目标尺寸
                - resize 模式: 直接缩放到 input_size×input_size
                - crop 模式: 短边缩放到 input_size（如 256）
            crop_size: int, 裁剪尺寸（用于 crop 模式），默认等于 input_size
                - crop 模式: 中央裁剪到 crop_size×crop_size（如 224）
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
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 构建 transform
        self.model = self._build_transform()

    def _build_transform(self):
        """根据模式构建对应的 transform"""
        if self.mode == "resize":
            # 模式1: 直接缩放
            transform_list = [
                transforms.Resize((self.input_size, self.input_size),
                                  interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        elif self.mode == "crop":
            # 模式2: 先缩放短边，再中央裁剪（ImageNet 标准）
            transform_list = [
                transforms.Resize(self.input_size,
                                  interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'resize' or 'crop'")

        return transforms.Compose(transform_list)

    def __call__(self, image):
        """
        Args:
            image: PIL.Image.Image 或 torch.Tensor
                - PIL: 直接处理
                - Tensor: 形状 (C, H, W) 或 (H, W, C)，dtype uint8 或 float

        Returns:
            torch.Tensor: shape (C, H, W) 或 (1, C, H, W), dtype float32/float16
        """
        # 如果输入是 numpy 数组，先转为 PIL
        import numpy as np
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image.astype('uint8'), 'RGB')

        # 执行 transform
        input_data = self.model(image)

        # 可选：FP16
        if self.half:
            input_data = input_data.half()

        # 可选：添加 batch 维度
        if self.batch:
            input_data = input_data.unsqueeze(0)

        return input_data
