# -*- coding: utf-8 -*-

"""
@Time    : 2024/9/7 15:31
@File    : runtime_bakcend.py
@Author  : zj
@Description: 
"""

import os
import numpy as np
from typing import Dict, Any, List, Union, Type

from numpy import floating, signedinteger, bool_, unsignedinteger

from core.backends.backend_base import BackendBase


def _get_available_providers() -> List[str]:
    """获取系统中可用的 ONNX Runtime 执行提供者。"""
    import onnxruntime
    try:
        return onnxruntime.get_available_providers()
    except Exception as e:
        print(f"Warning: Could not get available providers: {e}")
        return ['CPUExecutionProvider']  # 保守默认


def _onnx_type_to_numpy(onnx_type: str) -> Type[bool_]:
    """
    将 ONNX 类型字符串转换为 NumPy 数据类型。
    Args:
        onnx_type (str): ONNX 类型，如 'tensor(float)', 'tensor(int64)'。
    Returns:
        对应的 NumPy dtype。
    """
    # 这是一个简化的映射，可以根据需要扩展
    type_map = {
        'tensor(float)': np.float32,
        'tensor(float16)': np.float16,
        'tensor(double)': np.float64,
        'tensor(int64)': np.int64,
        'tensor(int32)': np.int32,
        'tensor(int16)': np.int16,
        'tensor(int8)': np.int8,
        'tensor(uint8)': np.uint8,
        'tensor(bool)': np.bool_,
        # ... 添加其他需要的类型
    }
    dtype = type_map.get(onnx_type)
    if dtype is None:
        raise ValueError(f"Unsupported ONNX type: {onnx_type}. Please add it to the mapping.")
    return dtype


class BackendRuntime(BackendBase):
    """
    ONNX Runtime 推理后端实现。
    支持 CPU 和 GPU (CUDA, DirectML) 执行提供者。
    """

    def __init__(self, model_path: str, providers=None, **kwargs):
        """
        Args:
            model_path (str): .onnx 模型文件路径。
            providers (List[str]): 执行提供者列表，如 ['CUDAExecutionProvider', 'CPUExecutionProvider']。
                                   如果为 None，则自动选择 (优先 GPU)。
            **kwargs: 传递给 onnxruntime.InferenceSession 的其他参数。
        """
        super().__init__(model_path, **kwargs)
        self.providers = providers
        self.session = None
        # 将在 load 时初始化
        self.input_names: List[str] = []
        self.input_dtypes: Dict[str, np.dtype] = {}
        self.input_shapes: Dict[str, List[int]] = {}
        self.output_names: List[str] = []
        self.metadata: Dict[str, str] = {}

    def _load_model(self, **kwargs) -> Any:
        """
        加载 ONNX 模型并初始化会话。
        Returns:
            onnxruntime.InferenceSession 对象。
        """
        # 1. 验证模型文件
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # 2. 确定执行提供者
        available_providers = _get_available_providers()
        if self.providers is None:
            # 自动选择: 优先 CUDA, 然后 CPU
            self.providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else [
                'CPUExecutionProvider']
        else:
            # 验证请求的提供者是否可用
            for provider in self.providers:
                if provider not in available_providers:
                    print(f"Warning: Provider '{provider}' not available. Available: {available_providers}")
            # 过滤出可用的提供者
            self.providers = [p for p in self.providers if p in available_providers]
            if not self.providers:
                raise RuntimeError(
                    f"No requested providers are available. Requested: {self.providers}, Available: {available_providers}")

        print(f"Loading {self.model_path} with providers: {self.providers}")

        # 3. 创建 InferenceSession
        import onnxruntime
        try:
            session = onnxruntime.InferenceSession(self.model_path, providers=self.providers, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to create ONNX Runtime session: {e}") from e

        # 4. 提取模型信息
        self._extract_model_info(session)

        return session

    def _extract_model_info(self, session: Any):
        """从 ONNX 会话中提取输入/输出信息和元数据。"""
        # --- 输入信息 ---
        self.input_names = [inp.name for inp in session.get_inputs()]
        self.input_dtypes = {inp.name: _onnx_type_to_numpy(inp.type) for inp in session.get_inputs()}
        self.input_shapes = {inp.name: inp.shape for inp in session.get_inputs()}

        # --- 输出信息 ---
        self.output_names = [out.name for out in session.get_outputs()]

        # --- 元数据 ---
        try:
            meta = session.get_modelmeta()
            self.metadata = meta.custom_metadata_map if meta.custom_metadata_map else {}
        except Exception as e:
            print(f"Warning: Could not extract model metadata: {e}")
            self.metadata = {}

        # --- 日志 ---
        print(f"\n=== ONNX Model Info ===")
        print(
            f"  Inputs: {dict(zip(self.input_names, [(self.input_shapes[n], self.input_dtypes[n]) for n in self.input_names]))}")
        print(f"  Outputs: {self.output_names}")
        print(f"  Metadata: {self.metadata}")
        print(f"  Providers: {self.providers}")
        print(f"=========================\n")

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行 ONNX 推理。
        Args:
            input_data: 输入数据字典。
        Returns:
            输出数据字典。
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # --- 输入验证 (可选但推荐) ---
        for name in self.input_names:
            if name not in input_data:
                raise ValueError(f"Missing input: '{name}'. Required inputs: {self.input_names}")
            # 可以添加形状和类型检查 (如果性能允许)

        # --- 执行推理 ---
        try:
            # ONNX Runtime 的 run 方法返回一个输出列表
            outputs = self.session.run(self.output_names, input_data)
            # 转换为字典
            output_dict = dict(zip(self.output_names, outputs))
            return output_dict
        except Exception as e:
            raise RuntimeError(f"ONNX Runtime inference failed: {e}") from e

    def get_input_names(self) -> List[str]:
        if not self._is_loaded:
            # 如果模型未加载，尝试从文件中提取 (可选，较复杂)
            # 这里简单返回空列表或抛出异常
            raise RuntimeError("Model not loaded. Cannot get input names.")
        return self.input_names.copy()  # 返回副本

    def get_output_names(self) -> List[str]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Cannot get output names.")
        return self.output_names.copy()

    def get_input_dtypes(self) -> Dict[str, np.dtype]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Cannot get input dtypes.")
        return self.input_dtypes.copy()

    def get_input_shapes(self) -> Dict[str, List[int]]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Cannot get input shapes.")
        return self.input_shapes.copy()

    def get_metadata(self) -> Dict[str, str]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Cannot get metadata.")
        return self.metadata.copy()

    # def cleanup(self):
    #     """清理资源。ONNX Runtime 会话通常在销毁时自动清理，但显式调用更安全。"""
    #     if hasattr(self, 'session') and self.session is not None:
    #         # ONNX Runtime 没有显式的 session.close()，但可以尝试删除引用
    #         del self.session
    #         self.session = None
    #     self._is_loaded = False
    #     print(f"ONNX Runtime backend for {self.model_path} cleaned up.")
