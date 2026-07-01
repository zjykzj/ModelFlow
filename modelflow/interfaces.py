# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : interfaces.py
@Author  : zj
@Description: 核心抽象基类

定义所有组件的接口契约：
- InferencePipeline: 推理管线（Preprocessor + Backend + Postprocessor）
- BaseBackend: 推理后端
- BasePreprocessor / BasePostprocessor: 预处理 / 后处理
- BaseMetrics / BaseEvaluator: defined in eval/ module
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union
)

import numpy as np


# ==================== 1. InferencePipeline ====================


class InferencePipeline:
    """
    推理管线 = Preprocessor + Backend + Postprocessor

    数据流:
        image (HWC, BGR) → Preprocessor → tensor (NCHW) → Backend → raw (List[ndarray])
            ├── __call__: raw → Postprocessor → StructuredResult (dict)
            └── infer:     仅返回 raw（评估场景用）

    用法:
        pipeline = InferencePipeline(preprocessor, backend, postprocessor)
        result = pipeline(image, conf_thres=0.25, iou_thres=0.45)  # 端到端
        raw = pipeline.infer(tensor)  # 仅后端推理（评估循环内使用）
    """

    def __init__(self, preprocessor, backend, postprocessor):
        self.preprocessor = preprocessor
        self.backend = backend
        self.postprocessor = postprocessor

    def __call__(self, image: np.ndarray, **kwargs) -> Any:
        """端到端推理：预处理 → 后端推理 → 后处理

        Args:
            image: HWC BGR 图像
            **kwargs: 传给后处理的参数（conf_thres, iou_thres 等）

        Returns:
            结构化结果 (dict)，各任务格式不同
        """
        tensor = self.preprocessor(image)
        raw = self.backend(tensor)
        # 将原始图像尺寸注入后处理（用于坐标缩放）
        postproc_kwargs = dict(kwargs)
        if "original_shape" not in postproc_kwargs:
            postproc_kwargs["original_shape"] = image.shape[:2]
        return self.postprocessor(raw, **postproc_kwargs)

    def infer(self, tensor: np.ndarray) -> List[np.ndarray]:
        """仅后端推理（绕过预处理和后处理）

        评估场景下，Evaluator 已经通过 Dataset 拿到了预处理好的张量，
        只需要后端推理，然后自行收集原始输出。

        Args:
            tensor: NCHW float32 张量

        Returns:
            List[np.ndarray]: 后端原始输出列表
        """
        return self.backend(tensor)

    def warmup(self):
        """预热管线（各组件依次预热）"""
        if hasattr(self.preprocessor, "warmup"):
            self.preprocessor.warmup()
        if hasattr(self.backend, "warmup"):
            self.backend.warmup()
        if hasattr(self.postprocessor, "warmup"):
            self.postprocessor.warmup()


# ==================== 2. BaseBackend ====================


class BaseBackend(ABC):
    """推理后端基类

    职责：纯张量推理（预处理好的张量 → 原始输出）
    不含图像处理逻辑。

    子类必须实现 __call__ 方法。
    """

    def __init__(
        self,
        model_path: str,
        class_list: List[str],
        task_type: Optional[str] = None,
        half: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_path = model_path
        self.class_list = class_list
        self.task_type = task_type
        self.half = half
        self.device = device

    @abstractmethod
    def __call__(self, input_data: np.ndarray) -> List[np.ndarray]:
        """执行推理

        Args:
            input_data: NCHW float32 输入张量

        Returns:
            List[np.ndarray]: 原始输出列表
        """
        ...

    def warmup(self):
        """预热模型（默认空实现）"""
        pass

    def get_input_info(self) -> "ModelInfo":
        """获取输入信息"""
        raise NotImplementedError

    def get_output_info(self) -> List["ModelInfo"]:
        """获取输出信息"""
        raise NotImplementedError

    def print_model_info(self):
        """打印模型元信息"""
        info = [
            f"Model: {self.model_path}",
            f"Task:  {self.task_type}",
            f"Half:  {self.half}",
        ]
        try:
            info.append(f"Input: {self.get_input_info()}")
        except NotImplementedError:
            pass
        try:
            for o in self.get_output_info():
                info.append(f"Output: {o}")
        except NotImplementedError:
            pass
        print("\n".join(info))


# ==================== 3. Preprocessor / Postprocessor ====================


class BasePreprocessor(ABC):
    """预处理基类

    职责：图像 → 网络输入张量
    """

    @abstractmethod
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """预处理单张图像

        Args:
            image: HWC BGR 图像 (H, W, 3)
            **kwargs: 各预处理器特有参数

        Returns:
            NCHW float32 张量
        """
        ...


class BasePostprocessor(ABC):
    """后处理基类

    职责：后端原始输出 → 结构化结果
    """

    @abstractmethod
    def __call__(self, raw: List[np.ndarray], **kwargs) -> Any:
        """后处理

        Args:
            raw: 后端原始输出列表
            **kwargs: 后处理参数（conf_thres, iou_thres 等）

        Returns:
            结构化结果 (dict)，各任务格式不同
        """
        ...



