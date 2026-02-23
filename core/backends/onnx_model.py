# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/23 13:49
@File    : onnx_model.py
@Author  : zj
@Description: 
"""

import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)


@dataclass
class IOInfo:
    """输入/输出信息数据类"""
    name: str
    shape: List[int]
    dtype: np.dtype
    index: int


class ONNXModel:
    """
    统一的ONNX模型推理类
    支持分类、检测、分割三种任务类型
    """

    def __init__(
            self,
            model_path: str,
            class_list: List[str],
            label_list: Optional[List[str]] = None,
            half: bool = False,
            device: Optional[str] = None,
            providers: Optional[List[str]] = None,
            stride: int = 32,
    ):
        """
        初始化ONNX模型推理类

        Args:
            model_path: 模型文件路径
            class_list: 类别列表
            label_list: 标签列表（可选）
            half: 是否使用半精度（当前CPU模式下暂不生效）
            device: 设备类型（当前强制使用CPU）
            providers: 执行提供者列表（默认CPU）
            stride: 步长（主要用于分割任务）
        """
        self.model_path = model_path
        self.class_list = class_list
        self.label_list = label_list
        self.half = half
        self.device = device
        self.stride = stride

        logger.info(f"model_path: {model_path}")

        # 设置执行提供者（当前强制CPU，可根据需要扩展）
        if providers is None:
            providers = ['CPUExecutionProvider']

        # 初始化推理会话
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        logger.info(f"session: {self.session}")

        # 获取输入输出完整信息
        self._init_io_info()

        # 热身
        self.__warmup()

    def _init_io_info(self):
        """初始化输入输出信息（包含name, shape, dtype）"""
        # 获取输入信息
        inputs = self.session.get_inputs()
        self.input_infos: List[IOInfo] = []
        self.input_names: List[str] = []
        self.input_shapes: List[List[int]] = []
        self.input_dtypes: List[np.dtype] = []

        for idx, inp in enumerate(inputs):
            io_info = IOInfo(
                name=inp.name,
                shape=list(inp.shape),
                dtype=self._get_numpy_dtype(inp.type),
                index=idx
            )
            self.input_infos.append(io_info)
            self.input_names.append(inp.name)
            self.input_shapes.append(list(inp.shape))
            self.input_dtypes.append(io_info.dtype)

        # 获取输出信息
        outputs = self.session.get_outputs()
        self.output_infos: List[IOInfo] = []
        self.output_names: List[str] = []
        self.output_shapes: List[List[int]] = []
        self.output_dtypes: List[np.dtype] = []

        for idx, out in enumerate(outputs):
            io_info = IOInfo(
                name=out.name,
                shape=list(out.shape),
                dtype=self._get_numpy_dtype(out.type),
                index=idx
            )
            self.output_infos.append(io_info)
            self.output_names.append(out.name)
            self.output_shapes.append(list(out.shape))
            self.output_dtypes.append(io_info.dtype)

        logger.info(f"input_names: {self.input_names}")
        logger.info(f"input_shapes: {self.input_shapes}")
        logger.info(f"input_dtypes: {self.input_dtypes}")
        logger.info(f"output_names: {self.output_names}")
        logger.info(f"output_shapes: {self.output_shapes}")
        logger.info(f"output_dtypes: {self.output_dtypes}")

    def _get_numpy_dtype(self, onnx_dtype: str) -> np.dtype:
        """将ONNX数据类型转换为numpy数据类型"""
        dtype_map = {
            'tensor(float)': np.float32,
            'tensor(double)': np.float64,
            'tensor(int8)': np.int8,
            'tensor(int16)': np.int16,
            'tensor(int32)': np.int32,
            'tensor(int64)': np.int64,
            'tensor(uint8)': np.uint8,
            'tensor(uint16)': np.uint16,
            'tensor(uint32)': np.uint32,
            'tensor(uint64)': np.uint64,
            'tensor(bool)': np.bool_,
        }
        return dtype_map.get(onnx_dtype, np.float32)

    def __warmup(self):
        """模型热身，使用动态生成的dummy input"""
        try:
            dummy_inputs = []
            for shape in self.input_shapes:
                # 处理动态维度（将-1替换为1）
                processed_shape = [1 if dim == -1 else dim for dim in shape]
                dummy_input = np.random.randn(*processed_shape).astype(np.float32)
                dummy_inputs.append(dummy_input)

            for _ in range(3):
                feed_data = dict(zip(self.input_names, dummy_inputs))
                self.session.run(None, feed_data)

            logger.info("Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _prepare_input(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        准备输入数据，支持多种输入格式

        Args:
            input_data: 输入数据，支持以下格式：
                - np.ndarray: 单个输入数组
                - List[np.ndarray]: 多个输入数组列表
                - Dict[str, np.ndarray]: 命名输入字典

        Returns:
            Dict[str, np.ndarray]: 格式化的输入字典
        """
        if isinstance(input_data, dict):
            # 已经是字典格式，直接使用
            return input_data
        elif isinstance(input_data, list):
            # 列表格式，转换为字典
            if len(input_data) != len(self.input_names):
                raise ValueError(
                    f"Input list length ({len(input_data)}) doesn't match "
                    f"model input count ({len(self.input_names)})"
                )
            return dict(zip(self.input_names, input_data))
        elif isinstance(input_data, np.ndarray):
            # 单个数组，根据输入数量处理
            if len(self.input_names) == 1:
                return {self.input_names[0]: input_data}
            else:
                raise ValueError(
                    f"Model expects {len(self.input_names)} inputs, "
                    f"but got single array"
                )
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def __call__(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
            output_names: Optional[List[str]] = None,
            return_dict: bool = False
    ) -> Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray], np.ndarray]:
        """
        执行模型推理

        Args:
            input_data: 输入数据
            output_names: 指定输出名称列表（None表示返回所有输出）
            return_dict: 是否以字典形式返回输出

        Returns:
            根据task_type和参数返回不同格式：
            - classify: 默认返回logits数组 (N, num_classes)
            - detect: 返回所有输出元组
            - segment: 返回所有输出元组
            - return_dict=True: 返回 {output_name: output_data} 字典
        """
        # 准备输入
        feed_data = self._prepare_input(input_data)

        # 确定输出名称
        if output_names is None:
            output_names = self.output_names

        # 执行推理
        outputs = self.session.run(output_names, feed_data)

        # 根据task_type和return_dict处理输出
        if return_dict:
            return dict(zip(output_names, outputs))
        else:
            # 检测和分割任务返回所有输出
            return outputs if len(outputs) > 1 else outputs[0]

    def forward(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
            **kwargs
    ) -> Any:
        """前向传播（与__call__功能相同）"""
        return self.__call__(input_data, **kwargs)

    def infer(
            self,
            input_data: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
            **kwargs
    ) -> Any:
        """推理（与__call__功能相同）"""
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
        print(f"Model Path: {self.model_path}")
        print("=" * 60)
        print("INPUTS:")
        for i, info in enumerate(self.input_infos):
            print(f"  [{i}] name={info.name}, shape={info.shape}, dtype={info.dtype}")
        print("OUTPUTS:")
        for i, info in enumerate(self.output_infos):
            print(f"  [{i}] name={info.name}, shape={info.shape}, dtype={info.dtype}")
        print("=" * 60)
