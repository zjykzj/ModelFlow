# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 15:28
@File    : dataset.py
@Author  : zj
@Description: 
"""

import os
import cv2
from PIL import Image

from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)


def imread_pil(img_path):
    img = Image.open(img_path).convert('RGB')
    return img


def imread_cv(img_path, rgb=False):
    img = cv2.imread(img_path)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ClassifyDataset:

    def __init__(self, root, classes_list, label_list, imread=None):
        assert os.path.isdir(root), 'Path does not exist!'
        if classes_list is None:
            classes_list = os.listdir(root)

        self.root = root
        self.classes_list = classes_list
        self.label_list = label_list
        assert imread is not None
        self.imread = imread

        data_list = list()
        for cls_idx, (class_name, label_name) in enumerate(zip(self.classes_list, self.label_list)):
            label_dir = str(os.path.join(root, label_name))
            if not os.path.isdir(label_dir):
                continue
            img_name_list = os.listdir(label_dir)
            for img_name in img_name_list:
                img_path = os.path.join(label_dir, img_name)
                data_list.append([cls_idx, class_name, img_path])
        self.data_list = data_list
        logger.info(f"data_root is {root}")
        logger.info(f"data_list length is {len(data_list)}")

    def __getitem__(self, index):
        assert index < len(self.data_list), 'Index out of range!'
        cls_idx, class_name, img_path = self.data_list[index]

        img = self.imread(img_path)
        img_name = os.path.basename(img_path)

        return img, img_name, img_path, cls_idx, class_name

    def __len__(self):
        return len(self.data_list)
