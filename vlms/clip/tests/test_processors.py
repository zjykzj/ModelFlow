# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : test_processors.py
@Author  : zj
@Description: CLIP 处理器单元测试

覆盖: CLIPImagePreprocessor, CLIPPostprocessor

运行:
    pytest vlms/clip/tests/test_processors.py -v
"""

import pytest
import numpy as np

from vlms.clip import CLIPImagePreprocessor, CLIPPostprocessor


@pytest.fixture
def rgb_image():
    """创建一张虚拟 BGR 图像 (480, 640, 3)"""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


class TestCLIPProcessor:
    """CLIP 处理器测试"""

    def test_image_preprocess(self, rgb_image):
        preproc = CLIPImagePreprocessor(input_size=224)
        tensor = preproc(rgb_image)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_image_preprocess_values(self, rgb_image):
        """CLIP 使用不同的 Normalize 参数"""
        preproc = CLIPImagePreprocessor(input_size=224)
        tensor = preproc(rgb_image)
        # CLIP normalize 的均值/标准差不同
        assert -2.0 < tensor.mean() < 2.0

    def test_postprocess_similarity(self):
        postproc = CLIPPostprocessor(topk=3, class_list=["cat", "dog", "bird"])
        image_embed = np.random.randn(512).astype(np.float32)
        text_embed = np.random.randn(3, 512).astype(np.float32)
        result = postproc(image_embed, text_embed)
        assert "top_indices" in result
        assert len(result["top_indices"]) == 3
        assert "top_labels" in result
        assert result["top_probs"][0] >= result["top_probs"][-1]

    def test_postprocess_no_text_embed(self):
        """没有 text_embed 时返回 image_embed"""
        postproc = CLIPPostprocessor()
        image_embed = np.random.randn(512).astype(np.float32)
        result = postproc(image_embed)
        assert "image_embed" in result

    def test_postprocess_different_topk(self):
        postproc = CLIPPostprocessor(topk=5)
        image_embed = np.random.randn(512).astype(np.float32)
        text_embed = np.random.randn(10, 512).astype(np.float32)
        result = postproc(image_embed, text_embed)
        assert len(result["top_indices"]) == 5
