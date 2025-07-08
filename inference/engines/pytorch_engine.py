# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/8 14:05
@File    : pytorch_engine.py
@Author  : zj
@Description: 
"""

import torch


class PyTorchEngine:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None

    def load_model(self):
        """加载模型并移动到指定设备"""
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def get_device(self):
        return self.device

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        执行前向推理
        :param input_tensor: 输入张量 (NCHW)
        :return: 输出张量
        """
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
