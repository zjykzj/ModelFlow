# -*- coding: utf-8 -*-

"""
@date: 2021/7/3 下午3:21
@file: onnxruntime_demo.py
@author: zj
@description: use torchvision.transform to preprocess, and use onnxruntime to infer model
"""

import numpy as np
import onnxruntime
from PIL import Image

from zcls.config import cfg
from zcls.data.build import build_transform


def get_model(onnx_name):
    ort_session = onnxruntime.InferenceSession(onnx_name)
    return ort_session


def get_transform(cfg):
    transform, _ = build_transform(cfg, is_train=False)
    return transform


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def infer(data, transform, ort_session):
    transform_data = transform(data).unsqueeze(0)

    # compute ONNX Runtime output prediction
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    ort_inputs = {input_name: to_numpy(transform_data)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(input_name, output_name, ort_outs[0].shape)

    return ort_outs[0]


if __name__ == '__main__':
    cfg_file = 'configs/mobilenet_v1_224.yaml'
    onnx_name = 'outputs/mobilenet_v1_224.onnx'
    data_shape = (224, 224, 3)

    cfg.merge_from_file(cfg_file)
    transform = get_transform(cfg)
    model = get_model(onnx_name)

    data = Image.fromarray(np.ones(data_shape).astype(np.uint8))
    outputs = infer(data, transform, model)

    numpy_probs = softmax(outputs)

    # Calculate probability and output
    for i in range(10):
        print(outputs[0][i], numpy_probs[0][i])

    """
    1.0003108e-18 0.1
    4.820882e-19 0.1
    5.6452285e-19 0.1
    -8.1291454e-19 0.1
    -4.845873e-19 0.1
    -3.7140606e-19 0.1
    3.324726e-19 0.1
    1.4153383e-19 0.1
    -1.2045691e-19 0.1
    -1.0059446e-18 0.1
    """