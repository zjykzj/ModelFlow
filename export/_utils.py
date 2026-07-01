# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : utils.py
@Author  : zj
@Description: Preprocessing utility functions (NumPy implementation, zero external dependencies)

Provides image preprocessing pipelines for classification and detection models:
- letterbox: aspect-ratio-preserving resize + padding
- classify_preprocess: ImageNet standard preprocessing
- detect_preprocess: YOLO series standard preprocessing

This module replaces the dependency on the project root core/, making the export module self-contained.
"""

from typing import Tuple, Optional, Union
from pathlib import Path

import cv2
import numpy as np


# ==================== Constants ====================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ==================== LetterBox ====================


def letterbox(
    image: np.ndarray,
    target_size: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto_pad: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Aspect-ratio-preserving resize + padding. Returns (padded_image, scale, pad).

    Aligned with Ultralytics YOLO's LetterBox implementation.

    Args:
        image: HWC BGR image (H, W, 3)
        target_size: Target size. int for square, tuple for (w, h).
        color: Padding color (B, G, R).
        auto_pad: Minimum rectangle mode. True = pad only to stride(32) alignment
                  (efficient inference), False = always pad to target_size
                  (square, used for calibration data generation).

    Returns:
        padded_image: Image after padding.
        scale: (scale_w, scale_h) scaling factors.
        pad: (pad_left, pad_top) padding pixels on each side.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Compute scaling factor (take the minimum to keep the entire image visible)
    scale = min(target_w / w, target_h / h)
    scale_w, scale_h = scale, scale

    # Resize
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    if auto_pad:
        # Pad to multiples of 32 (YOLO stride requirement)
        pad_w = pad_w % 32
        pad_h = pad_h % 32

    # Distribute evenly to both sides
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Pad
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


# ==================== Classification Preprocessing ====================


def classify_preprocess(
    image: np.ndarray,
    crop_size: int = 224,
    resize_size: int = 256,
) -> np.ndarray:
    """Standard ImageNet classification model preprocessing pipeline.

    OpenCV BGR -> RGB -> Resize(256) -> CenterCrop(224) -> Normalize -> HWC->CHW

    Args:
        image: HWC BGR image.
        crop_size: Crop size (default 224).
        resize_size: Resize size (default 256).

    Returns:
        CHW float32 tensor (3, crop_size, crop_size), normalized.
    """
    h, w = image.shape[:2]

    # BGR -> RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize short side to resize_size
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

    # Normalize (ImageNet statistics)
    normalized = cropped.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW
    chw = np.transpose(normalized, (2, 0, 1))

    return chw


# ==================== Detection Preprocessing ====================


def detect_preprocess(
    image: np.ndarray,
    target_size: int = 640,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Standard YOLO detection/segmentation model preprocessing pipeline.

    LetterBox -> BGR->RGB -> /255 -> HWC->CHW

    Args:
        image: HWC BGR image.
        target_size: Target size (default 640).

    Returns:
        chw: CHW float32 tensor (3, target_size, target_size), range [0, 1].
        scale: (scale_w, scale_h) scaling factors.
        pad: (pad_left, pad_top) padding offsets.
    """
    # LetterBox (minimum rectangle mode, efficient inference)
    padded, scale, pad = letterbox(image, target_size, auto_pad=True)

    # BGR -> RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # HWC -> CHW
    chw = np.transpose(rgb, (2, 0, 1))
    # Divide by 255
    chw = chw.astype(np.float32) / 255.0

    return chw, scale, pad


# ==================== Calibration Data Generation Utilities ====================


def save_calib_data(
    data: np.ndarray,
    output_dir: Union[str, Path],
    filename: str,
    fmt: str = "bin",
) -> None:
    """Save preprocessed tensor as a calibration data file.

    Args:
        data: CHW float32 tensor.
        output_dir: Output directory.
        filename: Filename (without extension).
        fmt: Output format ("bin" or "npy").
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


# ==================== File Utilities ====================


def collect_images(
    input_dir: Union[str, Path],
    max_images: Optional[int] = None,
) -> list:
    """Collect all image files under a directory.

    Args:
        input_dir: Image directory.
        max_images: Maximum number of images to return (None means all).

    Returns:
        List of image file paths.
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
