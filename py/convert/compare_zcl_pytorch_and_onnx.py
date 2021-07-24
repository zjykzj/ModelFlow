# -*- coding: utf-8 -*-

"""
@date: 2021/7/24 下午1:34
@file: compare_zcl_pytorch_and_onnx.py
@author: zj
@description: compare zcls pytorch's result with onnxruntime's result
"""

import os
import torch
import argparse
import onnxruntime

import numpy as np
from functools import reduce

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.build import build_recognizer

from misc import to_numpy, softmax


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_size', type=int, default=224, help='Data Size. Default: 224')
    parser.add_argument('zcls_config_path', type=str, default=None, help='ZCls Config Path')
    parser.add_argument('onnx_name', type=str, default=None, help='Result ONNX Name')

    args = parser.parse_args()
    # print(args)
    return args


def get_zcls_model(cfg, device):
    model = build_recognizer(cfg, device=device)
    model.eval()
    return model


def get_zcls_result(model, data_shape, device):
    data_length = reduce(lambda x, y: x * y, data_shape)
    data = torch.arange(data_length).float().reshape(data_shape).to(device)

    torch_out = model(data)[KEY_OUTPUT]
    return data, torch_out


def get_onnx_model(onnx_name, data):
    ort_session = onnxruntime.InferenceSession(onnx_name)

    # compute ONNX Runtime output prediction
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    ort_inputs = {input_name: data}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs[0]


if __name__ == '__main__':
    args = parse_args()

    data_size = int(args.data_size)
    zcls_config_path = os.path.abspath(args.zcls_config_path)
    onnx_name = os.path.abspath(args.onnx_name)

    device = torch.device('cpu')
    cfg.merge_from_file(zcls_config_path)
    model = get_zcls_model(cfg, device)
    torch_data, torch_out = get_zcls_result(model, (1, 3, data_size, data_size), device)

    numpy_data = to_numpy(torch_data)
    numpy_out = get_onnx_model(onnx_name, numpy_data)

    np.testing.assert_allclose(to_numpy(torch_out), numpy_out, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    zcls_pytorch_probs = torch.softmax(torch_out, dim=1)
    onnxruntime_probs = softmax(numpy_out)
    assert np.allclose(to_numpy(zcls_pytorch_probs), onnxruntime_probs)
