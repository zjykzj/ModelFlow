# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_datasets.py
@Author  : zj
@Description: 数据集模块测试

覆盖: COCODetectionDataset, COCOSegmentDataset, ClassifyDataset
"""

import os
import tempfile
import pytest
import numpy as np

from modelflow.datasets.coco_detection import COCODetectionDataset
from modelflow.datasets.segment_dataset import COCOSegmentDataset
from modelflow.datasets.classify_dataset import ClassifyDataset


class TestCOCODetectionDataset:
    """COCO 检测数据集测试"""

    def test_empty_directory(self):
        """空目录返回长度为 0 的数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = COCODetectionDataset(tmpdir, class_list=["person"])
            assert len(ds) == 0

    def test_with_images(self):
        """有图片时返回正确长度"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 3 张虚拟图片
            for i in range(3):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(os.path.join(tmpdir, f"img_{i}.jpg"), img)

            ds = COCODetectionDataset(tmpdir, class_list=["person", "car"])
            assert len(ds) == 3

    def test_getitem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.jpg")
            import cv2
            cv2.imwrite(img_path, np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

            ds = COCODetectionDataset(tmpdir, class_list=["person"])
            image, gt = ds[0]
            assert image.shape == (100, 100, 3)
            assert "image_path" in gt
            assert "image_id" in gt

    def test_gt_json_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = COCODetectionDataset(tmpdir, class_list=[])
            assert ds.get_gt_json() == ""

    def test_gt_json_with_anno(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            anno_path = os.path.join(tmpdir, "anno.json")
            with open(anno_path, "w") as f:
                f.write("{}")
            ds = COCODetectionDataset(tmpdir, class_list=[], anno_json=anno_path)
            assert ds.get_gt_json() == anno_path


class TestClassifyDataset:
    """分类数据集测试"""

    def test_with_directory_structure(self):
        """按目录结构组织的分类数据集"""
        import cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 cat/ 和 dog/ 两个类别目录
            cat_dir = os.path.join(tmpdir, "cat")
            dog_dir = os.path.join(tmpdir, "dog")
            os.makedirs(cat_dir)
            os.makedirs(dog_dir)
            for i in range(3):
                cv2.imwrite(os.path.join(cat_dir, f"{i}.jpg"),
                            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
            for i in range(5):
                cv2.imwrite(os.path.join(dog_dir, f"{i}.jpg"),
                            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))

            ds = ClassifyDataset(tmpdir, class_list=["cat", "dog"])
            assert len(ds) == 8  # 3 + 5

    def test_getitem_class_id(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            cat_dir = os.path.join(tmpdir, "cat")
            os.makedirs(cat_dir)
            cv2.imwrite(os.path.join(cat_dir, "0.jpg"),
                        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))

            ds = ClassifyDataset(tmpdir, class_list=["cat", "dog"])
            img, gt = ds[0]
            assert gt["class_id"] == 0  # cat 是第一个类别
            assert gt["class_name"] == "cat"


class TestCOSegmentDataset:
    """COCO 分割数据集测试"""

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = COCOSegmentDataset(tmpdir, class_list=["person"])
            assert len(ds) == 0

    def test_with_images(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            cv2.imwrite(os.path.join(tmpdir, "img.jpg"),
                        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            ds = COCOSegmentDataset(tmpdir, class_list=["person"])
            assert len(ds) == 1

    def test_getitem(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmpdir:
            cv2.imwrite(os.path.join(tmpdir, "test.jpg"),
                        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            ds = COCOSegmentDataset(tmpdir, class_list=["person"])
            image, gt = ds[0]
            assert "image_id" in gt
