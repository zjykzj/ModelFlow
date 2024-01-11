# -*- coding: utf-8 -*-

"""
@date: 2024/1/7 下午9:43
@file: yolov8_base.py
@author: zj
@description:

Yolov8: https://github.com/ultralytics/ultralytics
Commit id: e58db228c2fd9856e7bff54a708bf5acde26fb29

"""

import os
import cv2
import copy

from tqdm import tqdm

import numpy as np
from numpy import ndarray

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from general import LOGGER, CLASSES_NAME
from yolov8_util import LetterBox


def pre_transform(im, imgsz, stride=32, pt=False):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = all(x.shape == im[0].shape for x in im)
    print(f"imgsz: {imgsz}")
    print(f"auto = {same_shapes and pt}")
    print(f"stride = {stride}")
    letterbox = LetterBox(imgsz, auto=same_shapes and pt, stride=stride)
    return [letterbox(image=x) for x in im]
