# -*- coding: utf-8 -*-

"""
@date: 2021/7/3 下午3:10
@file: pytorch_to_onnx.py
@author: zj
@description: Convert Pytorch model to onnx format
"""

import torch
import numpy as np
import onnx
import onnxruntime

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.build import build_recognizer


def get_model(cfg):
    device = torch.device('cpu')
    model = build_recognizer(cfg, device=device)
    model.eval()
    return model


def create_model():
    cfg.merge_from_file(config_path)
    model = get_model(cfg)
    # print(model)

    data = torch.ones(data_shape)
    torch_out = model(data)[KEY_OUTPUT]
    torch.onnx.export(model,
                      data,
                      onnx_name,
                      export_params=True,
                      verbose=False,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['inputs'],
                      output_names=['outputs'],
                      dynamic_axes={
                          'inputs': {0: "batch_size"},
                          'outputs': {0: "batch_size"}
                      })
    return data, torch_out


def check_model(verbose=False):
    # Load the ONNX model
    model = onnx.load(onnx_name)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))
    graph = onnx.helper.printable_graph(model.graph)
    if verbose:
        print(graph)


def compute(data, torch_out):
    ort_session = onnxruntime.InferenceSession(onnx_name)

    # compute ONNX Runtime output prediction
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    ort_inputs = {input_name: to_numpy(data)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(input_name, output_name, ort_outs[0].shape)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    return ort_outs[0]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


if __name__ == '__main__':
    data_shape = (1, 3, 224, 224)
    config_path = 'configs/mobilenet_v1_224.yaml'
    onnx_name = 'outputs/mobilenet_v1_224.onnx'

    data, torch_out = create_model()
    torch_probs = torch.softmax(torch_out, dim=1)
    print('torch_probs:', torch_probs)

    check_model()
    outputs = compute(data, torch_out)

    numpy_probs = softmax(outputs)
    assert np.allclose(to_numpy(torch_probs), numpy_probs)

    # Calculate probability and output
    for i in range(10):
        print(outputs[0][i], numpy_probs[0][i])
    """
    9.525564e-19 0.1
    4.7569627e-19 0.1
    4.789132e-19 0.1
    -9.205878e-19 0.1
    -5.4157593e-19 0.1
    -4.018159e-19 0.1
    2.8573277e-19 0.1
    -5.116766e-20 0.1
    -3.049115e-19 0.1
    -9.347719e-19 0.1
    """
