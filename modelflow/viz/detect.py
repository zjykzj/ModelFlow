# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : detect.py
@Author  : zj
@Description: 检测可视化

通过 DataFlow-CV 的 COCOVisualizer 实现。
"""

from typing import Optional

import cv2
import numpy as np

from modelflow.core.interfaces import BaseVisualizer


class DetectVisualizer(BaseVisualizer):
    """检测可视化（DataFlow-CV 桥接）

    Args:
        class_list: 类别名称列表（用于标签文字）
    """

    def __init__(self, class_list=None):
        self.class_list = class_list or []

    def draw(self, image: np.ndarray, prediction, **kwargs) -> np.ndarray:
        """在图像上绘制检测框、标签和置信度

        Args:
            image: HWC BGR 图像
            prediction: detect postprocessor 的输出 dict
            **kwargs: 绘制参数（颜色、线宽等）

        Returns:
            带标注的图像
        """
        result = image.copy()
        boxes = prediction.get("boxes", [])
        scores = prediction.get("scores", [])
        class_ids = prediction.get("class_ids", [])
        class_names = prediction.get("class_names", [])
        color = kwargs.get("color", (0, 255, 0))

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            label = f"{class_names[i] if class_names else class_ids[i]}: {scores[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(result, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        return result

    # DataFlow-CV 桥接（可选）
    @classmethod
    def from_dataflow_cv(cls, pred_json: str, image_dir: str, output_dir: str, **kwargs):
        """通过 DataFlow-CV 的 COCOVisualizer 可视化

        Args:
            pred_json: COCO 格式预测 JSON
            image_dir: 图片目录
            output_dir: 输出目录
            **kwargs: 参数透传

        Returns:
            COCOVisualizer 实例（dataflow.visualize.COCOVisualizer）
        """
        try:
            from dataflow.visualize import COCOVisualizer
            return COCOVisualizer(
                annotation_file=pred_json,
                image_dir=image_dir,
                is_save=True,
                output_dir=output_dir,
                **kwargs,
            )
        except ImportError:
            raise ImportError("DataFlow-CV not installed. Install: pip install dataflow-cv")
