# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/23 15:22
@File    : triton_model.py
@Author  : zj
@Description: NVIDIA Triton Inference Server 统一推理框架
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)

# 尝试导入 tritonclient，支持 gRPC 和 HTTP 两种协议
try:
    import tritonclient.grpc as grpcclient
    from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    logger.warning("tritonclient.grpc not available")

try:
    import tritonclient.http as httpclient
    from tritonclient.http import InferenceServerClient as HTTPClient
    from tritonclient.http import InferInput as HTTPInferInput
    from tritonclient.http import InferRequestedOutput as HTTPInferRequestedOutput

    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    logger.warning("tritonclient.http not available")


@dataclass
class IOInfo:
    """输入/输出信息数据类"""
    name: str
    shape: List[int]
    dtype: np.dtype
    index: int


class TritonModel:
    """
    统一的 Triton Inference Server 模型推理类
    支持分类、检测、分割三种任务类型
    接口设计与 ONNXModel/TRTModel 保持一致
    """

    def __init__(
            self,
            model_name: str,
            class_list: List[str],
            label_list: Optional[List[str]] = None,
            half: bool = False,
            device: Optional[str] = None,
            providers: Optional[List[str]] = None,
            stride: int = 32,
            server_url: str = 'localhost:8001',
            protocol: str = 'grpc',
            verbose: bool = False,
            ssl: bool = False,
            headers: Optional[Dict[str, str]] = None,
            timeout: float = None,
    ):
        self.model_name = model_name
        self.class_list = class_list
        self.label_list = label_list
        self.half = half
        self.device = device
        self.stride = stride
        self.server_url = server_url
        self.protocol = protocol.lower()
        self.verbose = verbose
        self.headers = headers
        self.timeout = timeout

        logger.info(f"model_name: {model_name}")
        logger.info(f"server_url: {server_url}")
        logger.info(f"protocol: {protocol}")

        # 初始化客户端
        self.client = self._create_client()

        # 获取模型配置和输入输出信息
        self._init_io_info()

        # 热身
        self.__warmup()

    def _create_client(self) -> Union[InferenceServerClient, HTTPClient]:
        """创建 Triton 客户端"""
        if self.protocol == 'grpc':
            if not GRPC_AVAILABLE:
                raise ImportError("tritonclient.grpc is required for gRPC protocol")
            client = grpcclient.InferenceServerClient(
                url=self.server_url,
                verbose=self.verbose
            )
        elif self.protocol == 'http':
            if not HTTP_AVAILABLE:
                raise ImportError("tritonclient.http is required for HTTP protocol")
            url = f"http://{self.server_url}" if '://' not in self.server_url else self.server_url
            client = httpclient.InferenceServerClient(url=url, verbose=self.verbose)
            if self.headers:
                client.update_headers(self.headers)
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}. Use 'grpc' or 'http'.")

        # 检查服务器健康状态
        if not client.is_server_live():
            raise ConnectionError(f"Triton server at {self.server_url} is not live")

        logger.info(f"Triton client connected successfully")
        return client

    def _init_io_info(self):
        """初始化输入输出信息（包含 name, shape, dtype）"""
        try:
            # 获取模型配置
            if self.protocol == 'grpc':
                metadata = self.client.get_model_metadata(self.model_name, as_json=True)
            else:
                metadata = self.client.get_model_metadata(self.model_name)

            # 解析输入信息
            self.input_infos: List[IOInfo] = []
            self.input_names: List[str] = []
            self.input_shapes: List[List[int]] = []
            self.input_dtypes: List[np.dtype] = []

            for idx, inp in enumerate(metadata.get('inputs', [])):
                name = inp.get('name') if isinstance(inp, dict) else inp.name
                shape_raw = inp.get('shape') if isinstance(inp, dict) else list(inp.shape)
                shape = [int(dim) for dim in shape_raw]
                dtype_str = inp.get('datatype') if isinstance(inp, dict) else inp.datatype
                dtype = self._get_numpy_dtype(dtype_str)

                io_info = IOInfo(name=name, shape=shape, dtype=dtype, index=idx)
                self.input_infos.append(io_info)
                self.input_names.append(name)
                self.input_shapes.append(shape)
                self.input_dtypes.append(dtype)

            # 解析输出信息
            self.output_infos: List[IOInfo] = []
            self.output_names: List[str] = []
            self.output_shapes: List[List[int]] = []
            self.output_dtypes: List[np.dtype] = []

            for idx, out in enumerate(metadata.get('outputs', [])):
                name = out.get('name') if isinstance(out, dict) else out.name
                shape_raw = out.get('shape') if isinstance(out, dict) else list(out.shape)
                shape = [int(dim) for dim in shape_raw]
                dtype_str = out.get('datatype') if isinstance(out, dict) else out.datatype
                dtype = self._get_numpy_dtype(dtype_str)

                io_info = IOInfo(name=name, shape=shape, dtype=dtype, index=idx)
                self.output_infos.append(io_info)
                self.output_names.append(name)
                self.output_shapes.append(shape)
                self.output_dtypes.append(dtype)

            logger.info(f"input_names: {self.input_names}")
            logger.info(f"input_shapes: {self.input_shapes}")
            logger.info(f"input_dtypes: {self.input_dtypes}")
            logger.info(f"output_names: {self.output_names}")
            logger.info(f"output_shapes: {self.output_shapes}")
            logger.info(f"output_dtypes: {self.output_dtypes}")

        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise

    def _get_numpy_dtype(self, triton_dtype: str) -> np.dtype:
        """将 Triton 数据类型转换为 numpy 数据类型"""
        dtype_map = {
            'FP32': np.float32,
            'FP64': np.float64,
            'INT8': np.int8,
            'INT16': np.int16,
            'INT32': np.int32,
            'INT64': np.int64,
            'UINT8': np.uint8,
            'UINT16': np.uint16,
            'UINT32': np.uint32,
            'UINT64': np.uint64,
            'BOOL': np.bool_,
            'BYTES': np.bytes_,
            'fp32': np.float32,
            'fp64': np.float64,
            'int8': np.int8,
            'int16': np.int16,
            'int32': np.int32,
            'int64': np.int64,
            'uint8': np.uint8,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'uint64': np.uint64,
            'bool': np.bool_,
        }
        return dtype_map.get(triton_dtype, np.float32)

    def _get_triton_dtype(self, np_dtype: np.dtype) -> str:
        """将 numpy 数据类型转换为 Triton 数据类型"""
        dtype_map = {
            np.float32: 'FP32',
            np.float64: 'FP64',
            np.int8: 'INT8',
            np.int16: 'INT16',
            np.int32: 'INT32',
            np.int64: 'INT64',
            np.uint8: 'UINT8',
            np.uint16: 'UINT16',
            np.uint32: 'UINT32',
            np.uint64: 'UINT64',
            np.bool_: 'BOOL',
        }
        return dtype_map.get(np_dtype, 'FP32')

    def __warmup(self):
        """模型热身"""
        try:
            dummy_inputs = []
            for shape in self.input_shapes:
                processed_shape = [1 if dim == -1 else dim for dim in shape]
                dummy_input = np.random.randn(*processed_shape).astype(np.float32)
                dummy_inputs.append(dummy_input)

            for _ in range(3):
                self.infer(input_data=dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs)

            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _prepare_input(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """准备输入数据，支持多种输入格式"""
        if isinstance(input_data, dict):
            input_list = []
            for name in self.input_names:
                if name not in input_data:
                    raise ValueError(f"Missing input: {name}")
                input_list.append(input_data[name])
            return input_list
        elif isinstance(input_data, list):
            if len(input_data) != len(self.input_names):
                raise ValueError(
                    f"Input list length ({len(input_data)}) doesn't match "
                    f"model input count ({len(self.input_names)})"
                )
            return input_data
        elif isinstance(input_data, np.ndarray):
            if len(self.input_names) == 1:
                return [input_data]
            else:
                raise ValueError(
                    f"Model expects {len(self.input_names)} inputs, "
                    f"but got single array"
                )
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _create_infer_inputs(
            self,
            input_data: List[np.ndarray]
    ) -> Union[List[InferInput], List[HTTPInferInput]]:
        """创建 Triton 推理输入对象"""
        inputs = []
        for i, (name, data) in enumerate(zip(self.input_names, input_data)):
            if self.protocol == 'grpc':
                infer_input = grpcclient.InferInput(name, list(data.shape), self._get_triton_dtype(data.dtype))
                infer_input.set_data_from_numpy(data)
            else:
                infer_input = httpclient.InferInput(name, list(data.shape), self._get_triton_dtype(data.dtype))
                infer_input.set_data_from_numpy(data)
            inputs.append(infer_input)
        return inputs

    def _create_infer_outputs(
            self,
            output_names: Optional[List[str]] = None
    ) -> Union[List[InferRequestedOutput], List[HTTPInferRequestedOutput]]:
        """创建 Triton 推理输出对象"""
        if output_names is None:
            output_names = self.output_names

        outputs = []
        for name in output_names:
            if self.protocol == 'grpc':
                outputs.append(grpcclient.InferRequestedOutput(name))
            else:
                outputs.append(httpclient.InferRequestedOutput(name))
        return outputs

    def __call__(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
            output_names: Optional[List[str]] = None,
            return_dict: bool = False,
            batch_size: int = 1,
            sequence_id: int = 0,
            priority: int = 0,
            timeout: float = None,
    ) -> Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray], np.ndarray]:
        """执行模型推理"""
        # ✅ 修复：使用 input_data 参数，让 infer() 内部处理转换
        outputs = self.infer(
            input_data=input_data,
            output_names=output_names,
            sequence_id=sequence_id,
            priority=priority,
            timeout=timeout or self.timeout
        )

        if return_dict:
            return outputs
        else:
            output_list = [outputs.get(name) for name in self.output_names if name in outputs]
            return output_list if len(output_list) > 1 else (output_list[0] if output_list else None)

    def infer(
            self,
            inputs: Optional[Union[List[InferInput], List[HTTPInferInput]]] = None,
            outputs: Optional[Union[List[InferRequestedOutput], List[HTTPInferRequestedOutput]]] = None,
            batch_size: int = 1,
            sequence_id: int = 0,
            priority: int = 0,
            timeout: float = None,
            input_data: Optional[Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]] = None,
            output_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """核心推理方法"""
        # 如果提供了 input_data 但没有 inputs，则创建 inputs
        if inputs is None and input_data is not None:
            input_list = self._prepare_input(input_data)
            inputs = self._create_infer_inputs(input_list)

        if inputs is None:
            raise ValueError("Either 'inputs' or 'input_data' must be provided")

        # 如果 outputs 为 None，使用 output_names 创建
        if outputs is None:
            outputs = self._create_infer_outputs(output_names)

        try:
            if self.protocol == 'grpc':
                response = self.client.infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs,
                    sequence_id=sequence_id,
                    priority=priority,
                    timeout=timeout
                )
                results = {}
                for output_name in self.output_names:
                    try:
                        # ✅ 修复：创建可写副本
                        arr = response.as_numpy(output_name)
                        results[output_name] = arr.copy() if arr is not None else None
                    except Exception:
                        pass
            else:
                response = self.client.infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs,
                    sequence_id=sequence_id,
                    priority=priority,
                    timeout=timeout
                )
                results = {}
                for output_name in self.output_names:
                    try:
                        # ✅ 修复：创建可写副本
                        arr = response.as_numpy(output_name)
                        results[output_name] = arr.copy() if arr is not None else None
                    except Exception:
                        pass

            return results

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    # def infer(
    #         self,
    #         inputs: Optional[Union[List[InferInput], List[HTTPInferInput]]] = None,
    #         outputs: Optional[Union[List[InferRequestedOutput], List[HTTPInferRequestedOutput]]] = None,
    #         batch_size: int = 1,
    #         sequence_id: int = 0,
    #         priority: int = 0,
    #         timeout: float = None,
    #         input_data: Optional[Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]] = None,
    #         output_names: Optional[List[str]] = None,
    # ) -> Dict[str, np.ndarray]:
    #     """核心推理方法"""
    #     # 如果提供了 input_data 但没有 inputs，则创建 inputs
    #     if inputs is None and input_data is not None:
    #         input_list = self._prepare_input(input_data)
    #         inputs = self._create_infer_inputs(input_list)
    #
    #     if inputs is None:
    #         raise ValueError("Either 'inputs' or 'input_data' must be provided")
    #
    #     # 如果 outputs 为 None，使用 output_names 创建
    #     if outputs is None:
    #         outputs = self._create_infer_outputs(output_names)
    #
    #     try:
    #         if self.protocol == 'grpc':
    #             response = self.client.infer(
    #                 model_name=self.model_name,
    #                 inputs=inputs,
    #                 outputs=outputs,
    #                 sequence_id=sequence_id,
    #                 priority=priority,
    #                 timeout=timeout
    #             )
    #             results = {}
    #             for output_name in self.output_names:
    #                 try:
    #                     results[output_name] = response.as_numpy(output_name)
    #                 except Exception:
    #                     pass
    #         else:
    #             response = self.client.infer(
    #                 model_name=self.model_name,
    #                 inputs=inputs,
    #                 outputs=outputs,
    #                 sequence_id=sequence_id,
    #                 priority=priority,
    #                 timeout=timeout
    #             )
    #             results = {}
    #             for output_name in self.output_names:
    #                 try:
    #                     results[output_name] = response.as_numpy(output_name)
    #                 except Exception:
    #                     pass
    #
    #         return results
    #
    #     except Exception as e:
    #         logger.error(f"Inference failed: {e}")
    #         raise

    def forward(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
            **kwargs
    ) -> Any:
        """前向传播"""
        return self.__call__(input_data, **kwargs)

    def get_input_info(self, index: int = 0) -> IOInfo:
        """获取指定输入的信息"""
        return self.input_infos[index]

    def get_output_info(self, index: int = 0) -> IOInfo:
        """获取指定输出的信息"""
        return self.output_infos[index]

    def print_model_info(self):
        """打印模型完整信息"""
        print("=" * 60)
        print(f"Model Name: {self.model_name}")
        print(f"Server URL: {self.server_url}")
        print(f"Protocol: {self.protocol}")
        print("=" * 60)
        print("INPUTS:")
        for i, info in enumerate(self.input_infos):
            print(f"  [{i}] name={info.name}, shape={info.shape}, dtype={info.dtype}")
        print("OUTPUTS:")
        for i, info in enumerate(self.output_infos):
            print(f"  [{i}] name={info.name}, shape={info.shape}, dtype={info.dtype}")
        print("=" * 60)

    def is_model_ready(self) -> bool:
        """检查模型是否就绪"""
        try:
            return self.client.is_model_ready(self.model_name)
        except Exception:
            return False

    def get_model_config(self) -> Dict:
        """获取模型配置"""
        try:
            if self.protocol == 'grpc':
                return self.client.get_model_config(self.model_name, as_json=True)
            else:
                return self.client.get_model_config(self.model_name)
        except Exception as e:
            logger.error(f"Failed to get model config: {e}")
            return {}

    def close(self):
        """关闭客户端连接"""
        try:
            self.client.close()
            logger.info("Triton client closed")
        except Exception as e:
            logger.warning(f"Failed to close client: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass
