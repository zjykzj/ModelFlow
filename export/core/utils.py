# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : utils.py
@Author  : zj
@Description: 预处理工具函数（NumPy 实现，零外部依赖）

提供分类和检测模型所需的图像预处理管线：
- letterbox: 保持宽高比缩放 + 填充
- classify_preprocess: ImageNet 标准预处理
- detect_preprocess: YOLO 系列标准预处理

此模块替代对项目根目录 core/ 的依赖，实现 export 模块自包含。
"""

from typing import Tuple, Optional, Union
from pathlib import Path

import cv2
import numpy as np


# ==================== 常量 ====================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ==================== LetterBox ====================


def letterbox(
    image: np.ndarray,
    target_size: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto_pad: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """保持宽高比的缩放 + 填充，返回 (padded_image, scale, pad)

    与 Ultralytics YOLO 的 LetterBox 实现对齐。

    Args:
        image: HWC BGR 图像 (H, W, 3)
        target_size: 目标尺寸，int 表示正方形，tuple 为 (w, h)
        color: 填充颜色 (B, G, R)
        auto_pad: 最小矩形模式。True=仅填充至 stride(32) 对齐（推理高效），
                  False=始终填充至 target_size（正方形，校准数据生成）

    Returns:
        padded_image: 填充后的图像
        scale: (scale_w, scale_h) 缩放因子
        pad: (pad_left, pad_top) 每边填充像素数
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    target_w, target_h = target_size
    h, w = image.shape[:2]

    # 计算缩放因子（取最小的，保证完整图像可见）
    scale = min(target_w / w, target_h / h)
    scale_w, scale_h = scale, scale

    # 缩放
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    if auto_pad:
        # 填充至能被 32 整除（YOLO 步长要求）
        pad_w = pad_w % 32
        pad_h = pad_h % 32

    # 均匀分配到两边
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # 填充
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return padded, (scale_w, scale_h), (pad_left, pad_top)


# ==================== 分类预处理 ====================


def classify_preprocess(
    image: np.ndarray,
    crop_size: int = 224,
    resize_size: int = 256,
) -> np.ndarray:
    """ImageNet 分类模型标准预处理管线

    OpenCV BGR → RGB → Resize(256) → CenterCrop(224) → Normalize → HWC→CHW

    Args:
        image: HWC BGR 图像
        crop_size: 裁剪尺寸（默认 224）
        resize_size: 缩放尺寸（默认 256）

    Returns:
        CHW float32 张量 (3, crop_size, crop_size)，已 Normalize
    """
    h, w = image.shape[:2]

    # BGR → RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize 短边到 resize_size
    if h < w:
        new_h = resize_size
        new_w = int(round(w * resize_size / h))
    else:
        new_w = resize_size
        new_h = int(round(h * resize_size / w))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # CenterCrop
    ch, cw = resized.shape[:2]
    start_h = (ch - crop_size) // 2
    start_w = (cw - crop_size) // 2
    cropped = resized[start_h : start_h + crop_size, start_w : start_w + crop_size]

    # Normalize（ImageNet 统计值）
    normalized = cropped.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW
    chw = np.transpose(normalized, (2, 0, 1))

    return chw


# ==================== 检测预处理 ====================


def detect_preprocess(
    image: np.ndarray,
    target_size: int = 640,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """YOLO 检测/分割模型标准预处理管线

    LetterBox → BGR→RGB → /255 → HWC→CHW

    Args:
        image: HWC BGR 图像
        target_size: 目标尺寸（默认 640）

    Returns:
        chw: CHW float32 张量 (3, target_size, target_size)，值域 [0, 1]
        scale: (scale_w, scale_h) 缩放因子
        pad: (pad_left, pad_top) 填充偏移量
    """
    # LetterBox（最小矩形模式，推理高效）
    padded, scale, pad = letterbox(image, target_size, auto_pad=True)

    # BGR → RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # HWC → CHW
    chw = np.transpose(rgb, (2, 0, 1))
    # 除以 255
    chw = chw.astype(np.float32) / 255.0

    return chw, scale, pad


# ==================== 校准数据生成工具 ====================


def save_calib_data(
    data: np.ndarray,
    output_dir: Union[str, Path],
    filename: str,
    fmt: str = "bin",
) -> None:
    """保存预处理后的张量为校准数据文件

    Args:
        data: CHW float32 张量
        output_dir: 输出目录
        filename: 文件名（不含扩展名）
        fmt: 输出格式 ("bin" 或 "npy")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{filename}.{fmt}"

    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)

    if fmt == "bin":
        data.tofile(out_file)
    else:
        np.save(out_file, data)


# ==================== 文件工具 ====================


def collect_images(
    input_dir: Union[str, Path],
    max_images: Optional[int] = None,
) -> list:
    """收集目录下的所有图片文件

    Args:
        input_dir: 图片目录
        max_images: 最大返回数量（None 表示全部）

    Returns:
        图片路径列表
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    input_path = Path(input_dir)
    if not input_path.exists():
        return []
    image_files = [
        p
        for p in input_path.iterdir()
        if p.suffix.lower() in extensions
    ]
    image_files.sort()

    if max_images is not None:
        image_files = image_files[:max_images]

    return image_files
