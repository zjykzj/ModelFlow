# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:05
@File    : onnxruntime_engine.py
@Author  : zj
@Description: 
"""

import onnxruntime as ort
import numpy as np


class ONNXRuntimeEngine:

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化 ONNX Runtime 推理引擎

        :param model_path: ONNX 模型文件路径 (.onnx)
        :param device: 推理设备 ('cuda' 或 'cpu')
        """
        self.model_path = model_path
        self.device = device
        self.session = None
        self.providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']

        # 检查设备是否可用
        available_providers = ort.get_available_providers()
        if self.providers[0] not in available_providers:
            print(f"警告：{device.upper()} 设备不可用，将使用 CPU 进行推理")
            self.providers = ['CPUExecutionProvider']

    def load_model(self):
        """加载 ONNX 模型"""
        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            print("模型加载成功！")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    def get_input_name(self):
        """获取模型输入名称"""
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("该方法只支持单输入模型")
        return inputs[0].name

    def get_output_names(self):
        """获取模型输出名称列表"""
        return [output.name for output in self.session.get_outputs()]

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        执行前向推理

        :param input_tensor: 输入张量 (NCHW), dtype=np.float32
        :return: 输出张量 (numpy)，如果是多输出则返回一个包含所有输出的列表
        """
        if not isinstance(input_tensor, np.ndarray):
            raise TypeError("输入必须是 numpy 数组")
        if input_tensor.dtype != np.float32:
            raise ValueError("输入数组的数据类型必须是 np.float32")

        input_name = self.get_input_name()
        output_names = self.get_output_names()

        try:
            outputs = self.session.run(output_names, {input_name: input_tensor})
        except Exception as e:
            raise RuntimeError(f"推理过程中出现错误: {e}")

        # 如果只有一个输出，则直接返回该输出；否则返回所有输出组成的列表
        return outputs[0] if len(outputs) == 1 else outputs
