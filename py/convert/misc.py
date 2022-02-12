# -*- coding: utf-8 -*-

"""
@date: 2021/7/24 下午1:32
@file: profile.py
@author: zj
@description: 
"""

import onnx

import numpy as np


def check_model(onnx_name, verbose=False):
    # Load the ONNX model
    model = onnx.load(onnx_name)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))
    graph = onnx.helper.printable_graph(model.graph)
    if verbose:
        print(graph)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
