# -*- coding: utf-8 -*-

"""
@date: 2022/2/12 下午1:46
@file: pytorch_to_onnx.py
@author: zj
@description: Convert pytorch model to onnx format
See:
1. [(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
2. [Converting to ONNX format](https://github.com/onnx/tutorials#converting-to-onnx-format)
"""

import onnx
import onnxruntime
import numpy as np

import torch.onnx

from tiny_net import Net


def load_model(weight_path):
    weights = torch.load(weight_path, map_location='cpu')
    model = Net()
    model.load_state_dict(weights)

    model.eval()
    return model


def check_onnx(onnx_path='pytorch.onnx'):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def check_output(x, torch_out, onnx_path='pytorch.onnx'):
    ort_session = onnxruntime.InferenceSession(onnx_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def export_to_onnx(torch_model, batch_size=1, img_dim=1, img_size=28, onnx_path="pytorch.onnx"):
    # Input to the model
    x = torch.randn(batch_size, img_dim, img_size, img_size, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    check_onnx(onnx_path=onnx_path)
    check_output(x, torch_out, onnx_path=onnx_path)


def main():
    weight_path = '../assets/mnist_cnn.pt'
    model = load_model(weight_path)

    export_to_onnx(model, batch_size=1, img_dim=1, img_size=28, onnx_path='../assets/mnist_cnn.onnx')


if __name__ == '__main__':
    main()
