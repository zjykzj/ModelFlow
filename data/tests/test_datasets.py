# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : test_datasets.py
@Author  : zj
@Description: data/ 模块单元测试

覆盖: COCODataset, ClassifyDataset, build_dataset, BaseDataset 接口
"""

import os
import tempfile
import pytest
import numpy as np

from data import BaseDataset, ClassifyDataset, COCODataset, build_dataset


class TestCOCODataset:
    """COCO 数据集测试"""

    def test_empty_directory(self):
        """空目录返回长度为 0 的数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = COCODataset(tmpdir, class_list=["person"])
            assert len(ds) == 0

    def test_with_images(self):
        """有图片时返回正确长度"""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(os.path.join(tmpdir, f"img_{i}.jpg"), img)

            ds = COCODataset(tmpdir, class_list=["person", "car"])
            assert len(ds) == 3

    def test_getitem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.jpg")
            import cv2
            cv2.imwrite(img_path, np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

            ds = COCODataset(tmpdir, class_list=["person"])
            image, gt = ds[0]
            assert image.shape == (100, 100, 3)
            assert "image_path" in gt
            assert "image_id" in gt

    def test_gt_json_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = COCODataset(tmpdir, class_list=[])
            assert ds.get_gt_json() == ""

    def test_gt_json_with_anno(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            anno_path = os.path.join(tmpdir, "anno.json")
            with open(anno_path, "w") as f:
                f.write("{}")
            ds = COCODataset(tmpdir, class_list=[], anno_json=anno_path)
            assert ds.get_gt_json() == anno_path

    def test_task_segment(self):
        """分割任务的 COCODataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = COCODataset(tmpdir, class_list=["person"], task="segment")
            assert ds.task == "segment"


class TestClassifyDataset:
    """分类数据集测试"""

    def test_with_directory_structure(self):
        """按目录结构组织的分类数据集"""
        import cv2
        with tempfile.TemporaryDirectory() as tmpdir:
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


class TestBuildDataset:
    """build_dataset 工厂函数测试"""

    def test_build_coco_from_name(self):
        """按名称构建 COCO 数据集（使用 temp path）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build_dataset("coco", path=tmpdir, val="", anno=None)
            assert isinstance(ds, COCODataset)
            assert ds.task == "detect"

    def test_build_coco_seg_from_name(self):
        """按名称构建 COCO 分割数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build_dataset("coco-seg", path=tmpdir, val="", anno=None)
            assert isinstance(ds, COCODataset)
            assert ds.task == "segment"

    def test_build_classify_from_name(self):
        """按名称构建分类数据集"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = build_dataset("imagenet", path=tmpdir, val="")
            assert isinstance(ds, ClassifyDataset)


class TestBaseDataset:
    """BaseDataset ABC 接口测试"""

    def test_abstract_interface(self):
        class MyDS(BaseDataset):
            def __len__(self): return 5
            def __getitem__(self, idx):
                return (np.zeros((10, 10, 3)), {"id": idx})
            def get_gt_json(self): return "/path/to/gt.json"

        ds = MyDS()
        assert len(ds) == 5
        img, gt = ds[0]
        assert img.shape == (10, 10, 3)
        assert gt["id"] == 0
        assert ds.get_gt_json() == "/path/to/gt.json"
