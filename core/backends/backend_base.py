# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/30 17:03
@File    : backend_base.py
@Author  : zj
@Description: 
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np


class BackendBase(ABC):
    """
    推理后端抽象基类。
    所有具体的后端 (ONNX, TensorRT, Triton) 必须继承此类。

    设计原则:
    - 统一使用 NumPy 数组作为输入/输出的交换格式，简化上层模型逻辑。
    - 提供上下文管理器 (with 语句) 支持，确保资源正确释放。
    - 提供模型元信息访问接口。
    """

    def __init__(self, model_path: str, **kwargs):
        """
        初始化后端，加载模型。
        Args:
            model_path (str): 模型文件路径。
            **kwargs: 后端特定的配置参数 (如 providers, device, precision 等)。
        """
        self.model_path = model_path
        self._is_loaded = False
        self.session = None
        # 子类应在 _load_model 中设置以下属性
        # self.input_names: List[str] = []
        # self.input_dtypes: Dict[str, np.dtype] = {}
        # self.input_shapes: Dict[str, List[int]] = {}
        # self.output_names: List[str] = []
        # self.metadata: Dict[str, str] = {}

    @abstractmethod
    def _load_model(self, **kwargs) -> Any:
        """
        子类必须实现此方法来加载具体的模型。
        Returns:
            加载好的模型对象或会话 (具体类型由子类决定)。
        """
        pass

    def load(self, **kwargs):
        """
        加载模型的公共入口。可以在此进行一些通用的初始化工作。
        Args:
            **kwargs: 传递给 _load_model 的额外参数。
        Raises:
            RuntimeError: 如果加载失败。
        """
        if self._is_loaded:
            print(f"Model {self.model_path} is already loaded. Skipping reload.")
            return

        print(f"Loading {self.model_path} ...")
        try:
            self.session = self._load_model(**kwargs)
            self._is_loaded = True
            # print(f"Model loaded successfully with {type(self.session).__name__}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_path}: {e}") from e

    @abstractmethod
    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行推理。
        Args:
            input_data: 输入数据字典，键为输入名称，值为 NumPy 数组。
                        数组的形状和数据类型必须与模型期望的输入匹配。
        Returns:
            输出数据字典，键为输出名称，值为 NumPy 数组。
        Raises:
            RuntimeError: 如果推理执行失败。
        """
        pass

    @abstractmethod
    def get_input_names(self) -> List[str]:
        """获取输入节点名称列表。"""
        pass

    @abstractmethod
    def get_output_names(self) -> List[str]:
        """获取输出节点名称列表。"""
        pass

    @abstractmethod
    def get_input_dtypes(self) -> Dict[str, np.dtype]:
        """获取输入节点名称到数据类型的映射。"""
        pass

    @abstractmethod
    def get_input_shapes(self) -> Dict[str, List[int]]:
        """获取输入节点名称到期望形状的映射 (可能包含动态维度)。"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, str]:
        """
        获取模型的元数据 (metadata)。
        Returns:
            字典，包含模型的自定义元信息 (如 author, version, task)。
        """
        pass

    def warmup(self, input_data: Optional[Dict[str, np.ndarray]] = None, runs: int = 10):
        """
        模型预热。执行多次推理以消除初始化开销，使后续推理时间测量更准确。
        Args:
            input_data: 用于预热的输入数据。如果为 None，则使用随机数据填充期望的输入形状。
            runs (int): 预热执行的次数。
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if input_data is None:
            # 自动生成随机输入数据
            input_data = {}
            input_shapes = self.get_input_shapes()
            input_dtypes = self.get_input_dtypes()
            for name in self.get_input_names():
                shape = input_shapes[name]
                dtype = input_dtypes[name]
                # 处理动态维度 (-1 或 None)，用 1 代替
                concrete_shape = [dim if dim > 0 else 1 for dim in shape]
                if np.issubdtype(dtype, np.floating):
                    input_data[name] = np.random.rand(*concrete_shape).astype(dtype)
                else:
                    input_data[name] = np.random.randint(0, 255, size=concrete_shape, dtype=dtype)

        print(f"Warming up {runs} times...")
        for _ in range(runs):
            _ = self.infer(input_data)
        print("Warmup completed.")

    def __enter__(self):
        """支持 with 语句。"""
        if not self._is_loaded:
            self.load()
        return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     """确保资源被清理。"""
    #     self.cleanup()
    #
    # @abstractmethod
    # def cleanup(self):
    #     """清理资源 (如 GPU 内存、会话、线程)。子类必须实现。"""
    #     self._is_loaded = False
    #
    # def __del__(self):
    #     """作为 __exit__ 的后备，确保对象销毁时资源被释放。"""
    #     if self._is_loaded:
    #         print(f"Warning: Backend for {self.model_path} was not properly cleaned up. Calling cleanup().")
    #         self.cleanup()

    # --- 可选的便捷方法 ---
    def get_input_shape(self, name: str) -> List[int]:
        """获取指定输入名称的期望形状。"""
        shapes = self.get_input_shapes()
        if name not in shapes:
            raise ValueError(f"Input name '{name}' not found.")
        return shapes[name]

    def get_input_dtype(self, name: str) -> np.dtype:
        """获取指定输入名称的期望数据类型。"""
        dtypes = self.get_input_dtypes()
        if name not in dtypes:
            raise ValueError(f"Input name '{name}' not found.")
        return dtypes[name]
