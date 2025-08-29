# -*- coding: utf-8 -*-

"""
@Time    : 2024/10/13 14:27
@File    : backend_triton.py
@Author  : zj
@Description:

pip3 install tritonclient[all] opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple

"""
import copy
from typing import List, Any

import numpy as np
from numpy import ndarray

import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


class BackendTriton:

    def __init__(self, triton_client, model_name, input_name, output_name, is_fp16=False):
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name

        if not triton_client.is_model_ready(model_name):
            triton_client.load_model(model_name)
            print("model loaded success: {}.".format(model_name))
        print(f"{str(triton_client.get_model_repository_index().models)}")

        self.triton_client = triton_client
        self.input_type = np.float16 if is_fp16 else np.float32
        print(f"input name: {self.input_name} - input type: {self.input_type}")

    def __call__(self, im: ndarray) -> List[Any]:
        # return super().infer(im)
        # Set input
        inputs = [grpcclient.InferInput(self.input_name, im.shape, np_to_triton_dtype(self.input_type))]
        inputs[0].set_data_from_numpy(im.astype(self.input_type))
        # Infer
        outputs = [grpcclient.InferRequestedOutput(self.output_name)]
        response = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        # Get output
        reshaped = [response.as_numpy(self.output_name)]
        # print(f"reshaped shape: {reshaped[0].shape}")
        return copy.deepcopy(reshaped)
