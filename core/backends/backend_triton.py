# -*- coding: utf-8 -*-

"""
@Time    : 2024/10/13 14:27
@File    : backend_triton.py
@Author  : zj
@Description:

pip3 install tritonclient[all] opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple

"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional
import logging
import queue
import threading

# Triton 客户端
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

from .backend_base import BackendBase

logger = logging.getLogger(__name__)


class BackendTriton(BackendBase):
    """
    Triton Inference Server 后端实现（支持连接池）。

    注意：
        - model_path: Triton 中注册的模型名称
        - 支持 'grpc' / 'http' 协议
        - 使用连接池复用客户端连接，提升高并发性能
    """

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path=model_path, **kwargs)

        # Triton 配置
        self.server_url: str = kwargs.get("server_url", "localhost:8001")
        self.model_version: str = kwargs.get("model_version", "1")
        self.protocol: str = kwargs.get("protocol", "grpc").lower()
        self.verbose: bool = kwargs.get("verbose", False)
        self.trust_ssl: bool = kwargs.get("trust_ssl", False)
        self.pool_size: int = kwargs.get("pool_size", 4)  # 新增：连接池大小

        # 连接池相关
        self.pool: queue.Queue = None
        self.pool_lock = threading.Lock()

        # 模型输入输出信息（以字典形式存储，便于查询）
        self.input_names: List[str] = []  # 输入张量名称列表
        self.output_names: List[str] = []  # 输出张量名称列表
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}  # 输入形状 {name: shape}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}  # 输出形状 {name: shape}
        self.input_dtypes: Dict[str, np.dtype] = {}  # 输入数据类型 {name: dtype}
        self.output_dtypes: Dict[str, np.dtype] = {}  # 输出数据类型 {name: dtype}
        self.metadata: Dict[str, str] = {}  # 自定义元数据

        # 类型绑定
        self.client_class = None
        self.input_class = None
        self.output_class = None

    def _load_model(self, **kwargs) -> Any:
        """
        初始化连接池 + 加载模型元数据（包括输入输出 shape/dtype）。
        """
        # 设置客户端类
        if self.protocol == "grpc":
            self.client_class = grpcclient.InferenceServerClient
            self.input_class = grpcclient.InferInput
            self.output_class = grpcclient.InferRequestedOutput
        elif self.protocol == "http":
            self.client_class = httpclient.InferenceServerClient
            self.input_class = httpclient.InferInput
            self.output_class = httpclient.InferRequestedOutput
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")

        temp_client = None
        try:
            temp_client = self.client_class(
                url=self.server_url,
                verbose=self.verbose,
                ssl=bool(self.trust_ssl),
            )

            if not temp_client.is_server_live():
                raise RuntimeError(f"Triton server at {self.server_url} is not live.")
            if not temp_client.is_model_ready(self.model_path, self.model_version):
                raise RuntimeError(f"Model '{self.model_path}' (v{self.model_version}) is not ready.")

            # === 1. 获取模型元数据（输入/输出名称、输入 shape）===
            model_metadata = temp_client.get_model_metadata(self.model_path, self.model_version)
            self.input_names.clear()
            for inp in model_metadata.inputs:
                self.input_names.append(inp.name)
                dtype = self._triton_dtype_to_numpy(inp.datatype)
                shape = tuple(inp.shape)  # 转为 tuple，统一风格
                self.input_dtypes[inp.name] = dtype
                self.input_shapes[inp.name] = shape

            self.output_names = [out.name for out in model_metadata.outputs]

            # === 2. 获取模型配置（输出 dtype 和 shape）===
            model_config = temp_client.get_model_config(self.model_path, self.model_version)
            config = model_config.config

            # 提取输出信息
            for output in config.output:
                name = output.name
                if name not in self.output_names:
                    continue  # 安全检查

                # 数据类型
                triton_dtype = output.data_type
                np_dtype = self._triton_dtype_to_numpy(triton_dtype)
                self.output_dtypes[name] = np_dtype

                # 形状：优先使用 reshape，否则用 dims
                if hasattr(output, "reshape") and output.reshape.shape:
                    # 使用 reshape.shape
                    shape = tuple(int(d) for d in output.reshape.shape)
                else:
                    # 否则使用原始 dims（注意：dims 是 int64 列表）
                    shape = tuple(int(d) if d > 0 else 1 for d in output.dims)  # -1 → 1（占位）
                self.output_shapes[name] = shape

            # 提取自定义 metadata
            if hasattr(config, "parameters") and config.parameters:
                self.metadata = {
                    k: v.string_value
                    for k, v in config.parameters.items()
                }
            else:
                self.metadata = {}

            temp_client.close()
            temp_client = None

        except Exception as e:
            if temp_client:
                temp_client.close()
            logger.error(f"Failed to load model metadata/config: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

        # === 3. 创建连接池 ===
        try:
            self.pool = queue.Queue(maxsize=self.pool_size)
            for _ in range(self.pool_size):
                client = self.client_class(
                    url=self.server_url,
                    verbose=self.verbose,
                    ssl=bool(self.trust_ssl),
                )
                self.pool.put(client)
            logger.info(f"Triton client connection pool created: size={self.pool_size}")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise RuntimeError(f"Failed to initialize connection pool: {e}") from e

        # === 4. 打印模型信息（统一风格）===
        self._print_model_info()

        self._is_loaded = True
        return None

    def _print_model_info(self):
        """打印模型的输入、输出、元数据等信息，格式与 BackendRuntime / BackendTensorRT 统一。"""
        print(f"\n=== Triton Model Info ===")
        print(f"Model Name:  {self.model_path}")
        print(f"Model Version: {self.model_version}")
        print(f"Server URL:  {self.server_url}")
        print(f"Protocol:    {self.protocol.upper()}")

        # --- 输入信息 ---
        print(f"\nInput(s):")
        if self.input_names:
            for name in self.input_names:
                shape = self.input_shapes[name]
                dtype = self.input_dtypes[name]
                print(f"  {name} ({dtype}): {list(shape)}")
        else:
            print("  <none>")

        # --- 输出信息 ---
        print(f"\nOutput(s):")
        if self.output_names:
            for name in self.output_names:
                dtype = self.output_dtypes.get(name, "unknown")
                shape = self.output_shapes.get(name, "?")
                dtype_str = str(dtype) if isinstance(dtype, np.dtype) else dtype
                shape_repr = list(shape) if isinstance(shape, (list, tuple)) else shape
                print(f"  {name} ({dtype_str}): {shape_repr}")
        else:
            print("  <none>")

        # --- 元数据 ---
        if self.metadata:
            print(f"\nMetadata: {dict(self.metadata)}")
        else:
            print(f"\nMetadata: <empty>")

        print(f"=========================\n")

    def _get_client_from_pool(self):
        """从池中获取 client，支持阻塞等待"""
        return self.pool.get(block=True)

    def _return_client_to_pool(self, client):
        """归还 client 到池"""
        try:
            self.pool.put_nowait(client)
        except queue.Full:
            # 池已满，安全关闭
            try:
                client.close()
            except:
                pass

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        从连接池获取 client，执行推理，归还 client。
        自动校验并转换输入 dtype 以匹配模型期望。
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        client = None
        try:
            client = self._get_client_from_pool()

            # 构造输入
            triton_inputs = []
            for name in self.input_names:
                if name not in input_data:
                    raise ValueError(f"Missing input tensor: {name}")
                array = input_data[name]

                # === 类型校验与自动转换 ===
                expected_dtype = self.input_dtypes[name]
                if array.dtype != expected_dtype:
                    logger.debug(
                        f"[Triton Backend] Input '{name}' dtype mismatch: "
                        f"got {array.dtype}, required {expected_dtype}. Converting automatically."
                    )
                    array = array.astype(expected_dtype)

                # === 形状校验（可选增强）===
                expected_shape = self.input_shapes[name]
                if len(array.shape) != len(expected_shape):
                    logger.debug(
                        f"[Triton Backend] Input '{name}' rank mismatch: "
                        f"got shape {array.shape}, expected rank {len(expected_shape)}."
                    )
                # 注意：Triton 支持动态 batch，所以只做 dtype 强约束

                triton_dtype = np_to_triton_dtype(array.dtype)
                inp = self.input_class(name, array.shape, triton_dtype)
                inp.set_data_from_numpy(array)
                triton_inputs.append(inp)

            # 构造输出
            triton_outputs = [self.output_class(name) for name in self.output_names]

            # 执行推理
            response = client.infer(
                model_name=self.model_path,
                model_version=self.model_version,
                inputs=triton_inputs,
                outputs=triton_outputs,
            )

            # 提取结果
            output_data = {name: response.as_numpy(name) for name in self.output_names}
            return output_data

        except Exception as e:
            raise RuntimeError(f"Triton inference failed: {e}") from e

        finally:
            if client:
                self._return_client_to_pool(client)

    # === 元数据查询接口（无需 client） ===
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

    # === 资源清理 ===
    def cleanup(self):
        """关闭所有池中连接"""
        if self.pool:
            while not self.pool.empty():
                try:
                    client = self.pool.get_nowait()
                    client.close()
                except:
                    pass
            self.pool = None
        self._is_loaded = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        if self._is_loaded:
            logger.debug(f"Backend for {self.model_path} not cleaned up properly.")
            self.cleanup()

    @staticmethod
    def _triton_dtype_to_numpy(triton_dtype: str | int) -> np.dtype:
        # 先处理整数枚举 → 字符串
        if isinstance(triton_dtype, int):
            int_to_str = {
                0: "INVALID",
                1: "BOOL",
                2: "UINT8",
                3: "UINT16",
                4: "UINT32",
                5: "UINT64",
                6: "INT8",
                7: "INT16",
                8: "INT32",
                9: "INT64",
                10: "FP16",
                11: "FP32",
                12: "FP64",
                13: "STRING",
                14: "BF16",  # ← 可能出现
                15: "BYTES",
            }
            if triton_dtype not in int_to_str:
                raise ValueError(f"Unsupported Triton data type enum: {triton_dtype}")
            triton_dtype = int_to_str[triton_dtype]

        # 再映射到 NumPy dtype
        if triton_dtype == "FP32":
            return np.float32
        elif triton_dtype == "FP16":
            return np.float16
        elif triton_dtype == "FP64":
            return np.float64
        elif triton_dtype == "INT8":
            return np.int8
        elif triton_dtype == "INT16":
            return np.int16
        elif triton_dtype == "INT32":
            return np.int32
        elif triton_dtype == "INT64":
            return np.int64
        elif triton_dtype == "UINT8":
            return np.uint8
        elif triton_dtype == "UINT16":
            return np.uint16
        elif triton_dtype == "UINT32":
            return np.uint32
        elif triton_dtype == "UINT64":
            return np.uint64
        elif triton_dtype == "BOOL":
            return np.bool_
        elif triton_dtype == "BYTES":
            return np.bytes_
        elif triton_dtype == "STRING":
            return np.object_
        elif triton_dtype == "BF16":
            # 关键：避免使用 np.dtype('bfloat16')
            # 改为返回 float32 作为兼容传输类型
            return np.float32
        else:
            raise ValueError(f"Unsupported Triton data type string: {triton_dtype}")
