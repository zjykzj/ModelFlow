# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : triton.py
@Author  : zj
@Description: Triton Inference Server 推理后端

通过 gRPC 或 HTTP 协议与 Triton Server 通信。

用法:
    backend = TritonBackend("Detect_COCO_YOLOv8s_TRT",
                            class_list=["person", "car"],
                            server_url="localhost:8001")
    outputs = backend(input_tensor)
"""

from typing import List, Optional, Union

import numpy as np

from modelflow.interfaces import BaseBackend
from modelflow.types import ModelInfo
from utils.logger import get_logger

logger = get_logger("modelflow.backend.triton")


class TritonBackend(BaseBackend):
    """Triton Server 推理后端"""

    def __init__(
        self,
        model_path: str,
        class_list: List[str],
        task_type: Optional[str] = None,
        half: bool = False,
        device: Optional[str] = None,
        server_url: str = "localhost:8001",
        protocol: str = "grpc",
        model_version: str = "1",
        timeout: int = 60,
        **kwargs,
    ):
        # model_path 作为 Triton 的 model_name
        super().__init__(model_path, class_list, task_type, half, device, **kwargs)
        self.model_name = model_path
        self.server_url = server_url
        self.protocol = protocol
        self.model_version = model_version
        self.timeout = timeout

        self._init_client()
        self._init_io_info()

    def _init_client(self):
        """初始化 Triton 客户端"""
        if self.protocol == "grpc":
            try:
                import tritonclient.grpc as grpcclient
                self.client = grpcclient.InferenceServerClient(
                    url=self.server_url, verbose=False
                )
            except ImportError:
                raise ImportError(
                    "tritonclient[grpc] not installed. "
                    "Install: pip install tritonclient[grpc]"
                )
        else:
            try:
                import tritonclient.http as httpclient
                self.client = httpclient.InferenceServerClient(
                    url=self.server_url, verbose=False
                )
            except ImportError:
                raise ImportError(
                    "tritonclient[http] not installed. "
                    "Install: pip install tritonclient[http]"
                )

        if not self.client.is_model_ready(self.model_name):
            logger.warning(f"Model {self.model_name} not ready on {self.server_url}")

    def _check_server(self):
        """检查 Triton 服务器是否可达"""
        try:
            self.client.is_server_live()
        except Exception as e:
            raise ConnectionError(
                f"Triton server unreachable at {self.server_url}: {e}"
            ) from e

    def _init_io_info(self):
        """从 Triton 服务端获取模型元信息"""
        config = self.client.get_model_config(self.model_name, as_json=True)
        self._input_infos = []
        self._output_infos = []
        self._input_names = []
        self._output_names = []

        for inp in config["input"]:
            info = ModelInfo(
                name=inp["name"],
                shape=inp["dims"],
                dtype=self._parse_dtype(inp["data_type"]),
            )
            self._input_infos.append(info)
            self._input_names.append(inp["name"])

        for out in config["output"]:
            info = ModelInfo(
                name=out["name"],
                shape=out["dims"],
                dtype=self._parse_dtype(out["data_type"]),
            )
            self._output_infos.append(info)
            self._output_names.append(out["name"])

    @staticmethod
    def _parse_dtype(dtype_str: str) -> np.dtype:
        mapping = {
            "TYPE_FP32": np.float32,
            "TYPE_FP16": np.float16,
            "TYPE_INT32": np.int32,
            "TYPE_INT64": np.int64,
            "TYPE_INT8": np.int8,
            "TYPE_UINT8": np.uint8,
        }
        return mapping.get(dtype_str, np.float32)

    @staticmethod
    def _get_triton_dtype(np_dtype: np.dtype) -> str:
        mapping = {
            np.float32: "TYPE_FP32",
            np.float16: "TYPE_FP16",
            np.int32: "TYPE_INT32",
            np.int64: "TYPE_INT64",
            np.int8: "TYPE_INT8",
            np.uint8: "TYPE_UINT8",
        }
        return mapping.get(np_dtype, "TYPE_FP32")

    def __call__(self, input_data: np.ndarray) -> List[np.ndarray]:
        try:
            return self._infer(input_data)
        except Exception as e:
            # 检查是否是连接错误（服务器不可达）
            error_msg = str(e).lower()
            if "connect" in error_msg or "unreachable" in error_msg or "refused" in error_msg:
                raise ConnectionError(
                    f"Triton server unreachable at {self.server_url}: {e}"
                ) from e
            raise

    def _infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        if self.protocol == "grpc":
            import tritonclient.grpc as grpcclient

            infer_input = grpcclient.InferInput(
                self._input_names[0],
                input_data.shape,
                self._get_triton_dtype(input_data.dtype),
            )
            infer_input.set_data_from_numpy(input_data)

            infer_outputs = [
                grpcclient.InferRequestedOutput(name)
                for name in self._output_names
            ]

            try:
                response = self.client.infer(
                    model_name=self.model_name,
                    inputs=[infer_input],
                    outputs=infer_outputs,
                    timeout=self.timeout,
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "deadline" in str(e).lower():
                    raise TimeoutError(
                        f"Triton inference timeout after {self.timeout}s "
                        f"for model {self.model_name} at {self.server_url}"
                    ) from e
                raise

            result = [response.as_numpy(name) for name in self._output_names]
        else:
            import tritonclient.http as httpclient

            infer_input = httpclient.InferInput(
                self._input_names[0],
                input_data.shape,
                self._get_triton_dtype(input_data.dtype),
            )
            infer_input.set_data_from_numpy(input_data)

            try:
                response = self.client.infer(
                    model_name=self.model_name,
                    inputs=[infer_input],
                    outputs=[httpclient.InferRequestedOutput(name)
                             for name in self._output_names],
                    timeout=self.timeout,
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "deadline" in str(e).lower():
                    raise TimeoutError(
                        f"Triton inference timeout after {self.timeout}s "
                        f"for model {self.model_name} at {self.server_url}"
                    ) from e
                raise

            result = [response.as_numpy(name) for name in self._output_names]

        if len(result) != len(self._output_names):
            raise ValueError(
                f"Output tensor count mismatch: expected {len(self._output_names)}, "
                f"got {len(result)}"
            )
        return result

    def get_input_info(self) -> ModelInfo:
        return self._input_infos[0]

    def get_output_info(self) -> List[ModelInfo]:
        return self._output_infos

    def is_model_ready(self) -> bool:
        return self.client.is_model_ready(self.model_name)
