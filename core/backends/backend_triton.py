# -*- coding: utf-8 -*-

"""
@Time    : 2024/10/13 14:27
@File    : backend_triton.py
@Author  : zj
@Description:

pip3 install tritonclient[all] opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple

"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

# Triton 客户端（需安装：pip install tritonclient[all]）
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from .backend_base import BackendBase

logger = logging.getLogger(__name__)


class BackendTriton(BackendBase):
    """
    Triton Inference Server 后端实现。

    注意：
        - model_path 参数在此类中表示 Triton 服务器上的 **模型名称**。
        - 初始化时通过 kwargs 指定 server_url、model_version、protocol 等。
    """

    def __init__(self, model_path: str, **kwargs):
        """
        初始化 Triton 后端。

        Args:
            model_path (str): Triton 服务器上的模型名称（即 model_path 被用作 model_name）。
            **kwargs: 支持以下额外参数：
                - server_url (str): Triton 服务地址，如 'localhost:8001' (gRPC) 或 'localhost:8000' (HTTP)
                - model_version (str): 模型版本，默认 "1"
                - protocol (str): 'grpc' 或 'http'，默认 'grpc'
                - verbose (bool): 是否启用详细日志，默认 False
                - trust_ssl (bool): 是否信任 SSL（仅 HTTP），默认 False
        """
        super().__init__(model_path=model_path, **kwargs)

        # Triton 配置
        self.server_url: str = kwargs.get("server_url", "localhost:8001")
        self.model_version: str = kwargs.get("model_version", "1")
        self.protocol: str = kwargs.get("protocol", "grpc").lower()
        self.verbose: bool = kwargs.get("verbose", False)
        self.trust_ssl: bool = kwargs.get("trust_ssl", False)

        # 客户端和会话
        self.client = None
        self.session = None  # 在 Triton 中，session 即 client + model 绑定

        # 缓存模型信息
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_dtypes: Dict[str, np.dtype] = {}
        self.input_shapes: Dict[str, List[int]] = {}
        self.metadata: Dict[str, str] = {}

    def _load_model(self, **kwargs) -> Any:
        """
        连接到 Triton 服务器并加载模型元数据。
        """
        try:
            # 创建客户端
            if self.protocol == "grpc":
                client_class = grpcclient.InferenceServerClient
            elif self.protocol == "http":
                client_class = httpclient.InferenceServerClient
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")

            self.client = client_class(
                url=self.server_url,
                verbose=self.verbose,
                ssl=bool(self.trust_ssl),
            )

            # 检查服务器是否存活
            if not self.client.is_server_live():
                raise RuntimeError(f"Triton server at {self.server_url} is not live.")

            # 检查模型是否就绪
            if not self.client.is_model_ready(self.model_path, self.model_version):
                raise RuntimeError(
                    f"Model '{self.model_path}' (version {self.model_version}) is not ready on Triton server."
                )

            # 获取模型元数据（返回的是 Protobuf 对象，不是 dict）
            model_metadata = self.client.get_model_metadata(self.model_path, self.model_version)

            # 正确解析 inputs
            self.input_names = []
            for inp in model_metadata.inputs:
                self.input_names.append(inp.name)
                dtype = self._triton_dtype_to_numpy(inp.datatype)
                shape = list(inp.shape)  # shape 是 RepeatedScalarContainer，转 list
                self.input_dtypes[inp.name] = dtype
                self.input_shapes[inp.name] = shape

            # 解析 outputs
            self.output_names = [out.name for out in model_metadata.outputs]

            # 获取模型配置（用于 metadata）
            model_config = self.client.get_model_config(self.model_path, self.model_version)

            # model_config 是 ModelConfigResponse 对象
            # 提取 parameters 字段作为 metadata
            if hasattr(model_config.config, "parameters") and model_config.config.parameters:
                self.metadata = {
                    key: val.string_value
                    for key, val in model_config.config.parameters.items()
                }
            else:
                self.metadata = {}

            return self.client

        except Exception as e:
            logger.error(f"Failed to load Triton model '{self.model_path}': {str(e)}")
            raise RuntimeError(f"Failed to connect to Triton or load model: {e}") from e

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        执行推理。

        Args:
            input_data: 输入数据字典，键为输入名称，值为 NumPy 数组。

        Returns:
            输出数据字典，键为输出名称，值为 NumPy 数组。

        Raises:
            RuntimeError: 如果推理失败。
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # 构建 Triton 输入张量
            triton_inputs = []
            for name in self.input_names:
                if name not in input_data:
                    raise ValueError(f"Missing input tensor: {name}")
                array = input_data[name]
                triton_dtype = np_to_triton_dtype(array.dtype)
                input_tensor = grpcclient.InferInput(name, array.shape, triton_dtype)
                input_tensor.set_data_from_numpy(array)
                triton_inputs.append(input_tensor)

            # 构建输出请求
            triton_outputs = []
            for name in self.output_names:
                triton_outputs.append(grpcclient.InferRequestedOutput(name))

            # 执行推理
            response = self.client.infer(
                model_name=self.model_path,
                model_version=self.model_version,
                inputs=triton_inputs,
                outputs=triton_outputs,
            )

            # 提取输出
            output_data = {}
            for name in self.output_names:
                output_data[name] = response.as_numpy(name)

            return output_data

        except Exception as e:
            raise RuntimeError(f"Triton inference failed: {e}") from e

    def get_input_names(self) -> List[str]:
        if not self._is_loaded:
            self.load()
        return self.input_names.copy()

    def get_output_names(self) -> List[str]:
        if not self._is_loaded:
            self.load()
        return self.output_names.copy()

    def get_input_dtypes(self) -> Dict[str, np.dtype]:
        if not self._is_loaded:
            self.load()
        return self.input_dtypes.copy()

    def get_input_shapes(self) -> Dict[str, List[int]]:
        if not self._is_loaded:
            self.load()
        return self.input_shapes.copy()

    def get_metadata(self) -> Dict[str, str]:
        if not self._is_loaded:
            self.load()
        return self.metadata.copy()

    # def cleanup(self):
    #     """清理资源：关闭 Triton 客户端连接。"""
    #     if self.client is not None:
    #         try:
    #             self.client.close()
    #         except Exception as e:
    #             logger.warning(f"Error closing Triton client: {e}")
    #         finally:
    #             self.client = None
    #             self.session = None
    #             self._is_loaded = False
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     """确保资源被清理。"""
    #     self.cleanup()
    #
    # def __del__(self):
    #     """作为 __exit__ 的后备，确保对象销毁时资源被释放。"""
    #     if self._is_loaded:
    #         logger.warning(f"Backend for {self.model_path} was not properly cleaned up. Calling cleanup().")
    #         self.cleanup()

    @staticmethod
    def _triton_dtype_to_numpy(triton_dtype: str) -> np.dtype:
        """
        将 Triton 数据类型字符串转换为 NumPy dtype。
        注意：这里做了简化映射，实际可扩展。
        """
        mapping = {
            "FP32": np.float32,
            "FP16": np.float16,
            "FP64": np.float64,
            "INT8": np.int8,
            "INT16": np.int16,
            "INT32": np.int32,
            "INT64": np.int64,
            "UINT8": np.uint8,
            "UINT16": np.uint16,
            "UINT32": np.uint32,
            "UINT64": np.uint64,
            "BOOL": np.bool_,
            "BYTES": np.bytes_,
        }
        dtype = mapping.get(triton_dtype)
        if dtype is None:
            raise ValueError(f"Unsupported Triton data type: {triton_dtype}")
        return dtype
