# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:09
@File    : onnxruntime_engine.py
@Author  : zj
@Description: 
"""

import numpy as np
from numpy import ndarray
from typing import Any, Dict, Optional, Union, Callable

from inference.base.engine import InferenceError  # 统一异常基类
from inference.base.base_detector import BaseDetector
from inference.detect.preprocessors.numpy_preprocessor import YOLOv5NumpyPreprocessor
from inference.detect.postprocessors.numpy_postprocessor import YOLOv5NumpyPostprocessor
from inference.engines.onnxruntime_engine import ONNXRuntimeEngine


class YOLOv5ONNXRuntimeEngine(BaseDetector):
    def __init__(
            self,
            model_path: str,
            device: str = "cuda",
            img_size: int = 640,
            preprocessor: Optional[Callable] = None,
            postprocessor: Optional[Callable] = None,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            class_names: Optional[list] = None,
    ):
        """
        初始化 YOLOv5 ONNX Runtime 推理引擎

        :param model_path: ONNX 模型路径 (.onnx)
        :param device: 推理设备 ('cuda' 或 'cpu')
        :param img_size: 输入图像尺寸 (默认 640)
        :param preprocessor: 自定义预处理器（可选）
        :param postprocessor: 自定义后处理器（可选）
        :param conf_thres: 置信度阈值
        :param iou_thres: NMS IOU 阈值
        :param class_names: 类别名称列表（可选，默认加载 coco.names）
        """
        super().__init__(
            model_path=model_path,
            device=device,
            preprocessor=preprocessor or YOLOv5NumpyPreprocessor(img_size=img_size),
            postprocessor=postprocessor or YOLOv5NumpyPostprocessor(
                img_size=img_size,
                conf_thres=conf_thres,
                iou_thres=iou_thres
            ),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            class_names=class_names
        )

        # 实例化底层 ONNX Runtime 引擎
        self.model_wrapper = ONNXRuntimeEngine(model_path, device=device)
        self.model_wrapper.load_model()

    def load_model(self) -> None:
        """模型已由构造函数加载，此方法保留用于接口一致性"""
        pass

    def infer(self, input_tensor: ndarray) -> ndarray:
        """
        执行 ONNX Runtime 推理
        :param input_tensor: 预处理后的输入张量
        :return: 模型输出张量
        """
        try:
            return self.model_wrapper.infer(input_tensor)
        except Exception as e:
            raise InferenceError(f"Model inference failed: {e}") from e

    def __call__(self, image: ndarray) -> Dict[str, ndarray]:
        """
        完整推理流程入口：preprocess -> infer -> postprocess
        """
        return super().__call__(image)
