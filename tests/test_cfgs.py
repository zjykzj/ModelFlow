# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_cfgs.py
@Author  : zj
@Description: 配置模块测试

覆盖: COCO 80 类, ImageNet 类
"""

from modelflow.cfgs.coco import class_list as coco_classes, NUM_CLASSES
from modelflow.cfgs.imagenet import get_imagenet_classes


class TestCocoCfg:
    """COCO 配置测试"""

    def test_num_classes(self):
        assert NUM_CLASSES == 80

    def test_class_list_length(self):
        assert len(coco_classes) == 80

    def test_class_names(self):
        assert "person" in coco_classes
        assert "car" in coco_classes
        assert "dog" in coco_classes

    def test_first_and_last(self):
        assert coco_classes[0] == "person"
        assert coco_classes[-1] == "toothbrush"


class TestImageNetCfg:
    """ImageNet 配置测试"""

    def test_get_classes(self):
        classes = get_imagenet_classes()
        assert len(classes) > 0
