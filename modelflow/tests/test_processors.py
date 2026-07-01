# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : test_processors.py
@Author  : zj
@Description: 处理器单元测试

覆盖: classify, detect, segment, semantic_seg 的 pre/post processors

运行:
    pytest modelflow/tests/test_processors.py -v
"""

import pytest
import numpy as np

from modelflow.processors.classify import ClassifyPreprocessor, ClassifyPostprocessor
from modelflow.processors.detect import DetectPreprocessor, DetectPostprocessor
from modelflow.processors.detect.ops import xywh2xyxy, nms, scale_boxes, clip_boxes
from modelflow.processors.segment import SegmentPreprocessor, SegmentPostprocessor
from modelflow.processors.semantic_seg import SemanticSegPreprocessor, SemanticSegPostprocessor

# ==================== Fixtures ====================

@pytest.fixture
def rgb_image():
    """创建一张虚拟 BGR 图像 (480, 640, 3)"""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


# ==================== Classify ====================

class TestClassifyProcessor:
    """分类处理器测试"""

    def test_preprocess_output_shape(self, rgb_image):
        preproc = ClassifyPreprocessor(crop_size=224)
        tensor = preproc(rgb_image)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_preprocess_larger_size(self, rgb_image):
        """大尺寸裁剪需要 resize_size 配套"""
        preproc = ClassifyPreprocessor(crop_size=384, resize_size=384)
        tensor = preproc(rgb_image)
        assert tensor.shape == (1, 3, 384, 384)

    def test_preprocess_values_normalized(self, rgb_image):
        preproc = ClassifyPreprocessor(crop_size=224)
        tensor = preproc(rgb_image)
        # Normalized values should be roughly centered around 0
        assert -3.0 < tensor.mean() < 3.0

    def test_postprocess_softmax(self):
        postproc = ClassifyPostprocessor(topk=3, class_list=["cat", "dog", "bird"])
        logits = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
        result = postproc([logits])
        assert len(result["class_ids"]) == 3
        assert len(result["scores"]) == 3
        assert len(result["class_names"]) == 3
        assert result["class_ids"][0] == 0  # cat has highest score
        assert result["scores"][0] > 0.5

    def test_postprocess_2d_logits(self):
        """处理 (num_classes,) 格式的 logits"""
        postproc = ClassifyPostprocessor(topk=2)
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = postproc([logits])
        assert len(result["class_ids"]) == 2

    def test_postprocess_no_class_list(self):
        """没有 class_list 时，不返回 class_names"""
        postproc = ClassifyPostprocessor(topk=2)
        logits = np.random.randn(1, 5).astype(np.float32)
        result = postproc([logits])
        assert "class_ids" in result
        assert "class_names" not in result


class TestDetectProcessor:
    """检测处理器测试"""

    def test_preprocess_letterbox_square(self, rgb_image):
        preproc = DetectPreprocessor(input_size=640)
        tensor = preproc(rgb_image)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_preprocess_value_range(self, rgb_image):
        preproc = DetectPreprocessor(input_size=640)
        tensor = preproc(rgb_image)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_different_size(self, rgb_image):
        preproc = DetectPreprocessor(input_size=416)
        tensor = preproc(rgb_image)
        assert tensor.shape == (1, 3, 416, 416)

    def test_postprocess_v8_format(self):
        """YOLOv8 格式 (1, 84, 8400)"""
        postproc = DetectPostprocessor(
            model_version="v8",
            class_list=["person", "car"],
            conf_thres=0.0,
            iou_thres=0.99,
            input_shape=(640, 640),
        )
        pred = np.zeros((1, 84, 8400), dtype=np.float32)
        pred[0, 4, 0] = 0.9  # one high-confidence detection
        result = postproc([pred], original_shape=(480, 640))
        assert "boxes" in result
        assert "scores" in result
        assert len(result["boxes"]) > 0

    def test_postprocess_v11_format(self):
        """YOLO11 格式与 v8 相同"""
        postproc = DetectPostprocessor(
            model_version="v11",
            conf_thres=0.0,
            iou_thres=0.99,
        )
        pred = np.zeros((1, 84, 8400), dtype=np.float32)
        pred[0, 4, 0] = 0.95
        result = postproc([pred])
        assert len(result["boxes"]) > 0

    def test_postprocess_empty_detection(self):
        """无检测时返回空数组"""
        postproc = DetectPostprocessor(conf_thres=0.5, iou_thres=0.5)
        pred = np.random.randn(1, 84, 8400).astype(np.float32) * 0.01
        result = postproc([pred])
        assert len(result["boxes"]) == 0
        assert len(result["scores"]) == 0

    def test_postprocess_with_class_names(self):
        """class_list 存在时返回 class_names"""
        postproc = DetectPostprocessor(
            class_list=["person", "car", "dog"],
            conf_thres=0.0,
            iou_thres=0.99,
        )
        pred = np.zeros((1, 7, 100), dtype=np.float32)  # 3 classes
        pred[0, 4, 0] = 0.9
        pred[0, 5, 1] = 0.8
        result = postproc([pred], original_shape=(640, 640))
        assert "class_names" in result

    def test_postprocess_coordinate_scaling(self, rgb_image):
        """坐标缩放应输出有效值（使用 YOLO 输出空间的坐标值）"""
        postproc = DetectPostprocessor(conf_thres=0.0, iou_thres=0.99,
                                        input_shape=(640, 640))
        pred = np.zeros((1, 6, 100), dtype=np.float32)
        pred[0, 4, 0] = 0.9
        pred[0, :4, 0] = [320, 320, 100, 100]  # center=(320,320) w=h=100
        result = postproc([pred], original_shape=(480, 640))
        assert result["boxes"][0, 2] > result["boxes"][0, 0]  # x2 > x1
        assert result["boxes"][0, 3] > result["boxes"][0, 1]  # y2 > y1


class TestDetectOps:
    """检测共享算子测试"""

    def test_xywh2xyxy(self):
        boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
        result = xywh2xyxy(boxes)
        assert result[0, 0] == 0.0   # x1 = 50 - 50
        assert result[0, 2] == 100.0  # x2 = 50 + 50

    def test_nms(self):
        boxes = np.array([
            [10, 10, 100, 100],
            [12, 12, 98, 98],    # high overlap with first
            [200, 200, 300, 300], # no overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_thresh=0.5)
        assert len(keep) == 2  # first and third
        assert keep[0] == 0
        assert keep[1] == 2

    def test_nms_single_box(self):
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        scores = np.array([0.9])
        keep = nms(boxes, scores, iou_thresh=0.5)
        assert len(keep) == 1

    def test_scale_boxes(self):
        boxes = np.array([[50, 50, 150, 150]], dtype=np.float32)
        result = scale_boxes((640, 640), boxes, (320, 320))
        assert result[0, 0] >= 0
        assert result[0, 2] <= 320

    def test_clip_boxes(self):
        boxes = np.array([[-10, -10, 700, 700]], dtype=np.float32)
        clip_boxes(boxes, (640, 640))
        assert boxes[0, 0] == 0
        assert boxes[0, 1] == 0
        assert boxes[0, 2] == 640
        assert boxes[0, 3] == 640


class TestSegmentProcessor:
    """分割处理器测试"""

    def test_preprocess_same_as_detect(self, rgb_image):
        """分割预处理与检测相同（LetterBox）"""
        seg_pre = SegmentPreprocessor(input_size=640)
        det_pre = DetectPreprocessor(input_size=640)
        seg_tensor = seg_pre(rgb_image)
        det_tensor = det_pre(rgb_image)
        assert seg_tensor.shape == det_tensor.shape

    def test_postprocess_v8_seg_format(self):
        """YOLOv8-seg 格式 (1, 116, 8400) + proto (1, 32, 160, 160)"""
        postproc = SegmentPostprocessor(conf_thres=0.0, iou_thres=0.99,
                                         input_shape=(640, 640))
        pred = np.zeros((1, 116, 100), dtype=np.float32)
        pred[0, 4, 0] = 0.9
        proto = np.random.randn(1, 32, 160, 160).astype(np.float32)
        result = postproc([pred, proto], original_shape=(480, 640))
        assert "boxes" in result
        assert "masks" in result or len(result["boxes"]) == 0

    def test_postprocess_no_proto(self):
        """没有 proto mask 时，返回不含 masks 的结果"""
        postproc = SegmentPostprocessor(conf_thres=0.0, iou_thres=0.99)
        pred = np.zeros((1, 116, 100), dtype=np.float32)
        pred[0, 4, 0] = 0.9
        result = postproc([pred], original_shape=(640, 640))
        assert "boxes" in result
        # masks 应为 None
        if len(result["boxes"]) > 0:
            assert result["masks"] is None


class TestSemanticSegProcessor:
    """语义分割处理器测试"""

    def test_preprocess_resize(self, rgb_image):
        preproc = SemanticSegPreprocessor(target_size=(224, 224))
        tensor = preproc(rgb_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_no_resize(self, rgb_image):
        """target_size=None 时保持原图尺寸"""
        preproc = SemanticSegPreprocessor(target_size=None)
        tensor = preproc(rgb_image)
        assert tensor.shape[2:] == (480, 640)

    def test_postprocess_argmax(self):
        postproc = SemanticSegPostprocessor()
        logits = np.random.randn(1, 21, 224, 224).astype(np.float32)
        result = postproc([logits])
        assert "class_map" in result
        assert result["class_map"].shape == (224, 224)
        assert result["class_map"].dtype == np.uint8

    def test_postprocess_2d_logits(self):
        """无 batch 维度的输入 (21, 224, 224)"""
        postproc = SemanticSegPostprocessor()
        logits = np.random.randn(21, 224, 224).astype(np.float32)
        result = postproc([logits])
        assert result["class_map"].shape == (224, 224)

