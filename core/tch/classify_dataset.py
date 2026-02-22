# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 15:28
@File    : dataset.py
@Author  : zj
@Description: 
"""

import os

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)


class ClassifyDataset(Dataset):

    def __init__(self, root, classes_list, label_list, imread=None, transform=None):
        assert os.path.isdir(root), 'Path does not exist!'

        self.root = root
        self.classes_list = classes_list
        self.label_list = label_list
        self.imread = imread
        self.transform = transform

        if self.classes_list is None:
            self.classes_list = os.listdir(root)

        data_list = list()
        for cls_idx, (class_name, label_name) in enumerate(zip(self.classes_list, self.label_list)):
            label_dir = os.path.join(root, label_name)
            if not os.path.isdir(label_dir):
                continue
            img_name_list = os.listdir(label_dir)
            for img_name in img_name_list:
                img_path = os.path.join(label_dir, img_name)
                data_list.append([cls_idx, class_name, img_path])
        self.data_list = data_list
        logger.info(f"data_root is {root}")
        logger.info(f"data_list len is {len(data_list)}")

    def __getitem__(self, index) -> T_co:
        # 返回图像名、标注的类别名和处理后的图像数
        assert index < len(self.data_list), 'Index out of range!'
        cls_idx, class_name, img_path = self.data_list[index]

        img = self.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)

        img_name = os.path.basename(img_path)
        return img, cls_idx, class_name, img_name, img_path

    def __len__(self):
        return len(self.data_list)

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)
