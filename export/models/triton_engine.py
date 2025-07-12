# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/10 16:37
@File    : triton_classifier.py
@Author  : zj
@Description: 
"""

import numpy as np
from typing import Any, Dict, List
import tritonclient.http as httpclient


class TritonClassifier:
    def __init__(
            self,
            model_name: str,
            server_url: str = "localhost:8000",
            input_name: str = "input",
            output_name: str = "output"
    ):
        """
        A classifier client for NVIDIA Triton Inference Server.

        :param model_name: Name of the deployed model on Triton
        :param server_url: URL of the Triton server
        :param input_name: Input tensor name
        :param output_name: Output tensor name
        """
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name

        self.client = httpclient.InferenceServerClient(url=server_url)

    def predict(self, input_data: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Send request to Triton server and get predictions.

        :param input_data: List of input tensors
        :return: Dictionary of output tensors
        """
        inputs = []
        for i, data in enumerate(input_data):
            input_tensor = httpclient.InferInput(self.input_name, data.shape, "FP32")
            input_tensor.set_data_from_numpy(data)
            inputs.append(input_tensor)

        outputs = [httpclient.InferRequestedOutput(self.output_name)]

        response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        result = response.get_response()

        return {
            name: response.as_numpy(name) for name in result["outputs"]
        }
