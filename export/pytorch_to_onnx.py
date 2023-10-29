# -*- coding: utf-8 -*-

"""
@date: 2022/2/12 下午1:46
@file: pytorch_to_onnx.py
@author: zj
@description: Convert pytorch model to onnx format
See:
1. [(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
2. [Converting to ONNX format](https://github.com/onnx/tutorials#converting-to-onnx-format)

Usage: Convert Pytorch ToyNet to ONNX:
    $ python pytorch_to_onnx.py

Usage: Convert Pytorch Resnet18 to ONNX:
    $ python pytorch_to_onnx.py --model resnet18 --save resnet18_pytorch.onnx

Usage: Dynamically setting batch size:
    $ python pytorch_to_onnx.py --model resnet18 --save resnet18_pytorch.onnx --dynamic

"""

import argparse
import numpy as np

import onnx
import onnxruntime

import torch.onnx
import torch.nn as nn
from torchvision import models

from toy_net import ToyNet


def parse_opt():
    parser = argparse.ArgumentParser(description="Pytorch to ONNX")
    parser.add_argument("--model", metavar="model", type=str, default=None,
                        help="Pytorch Model path, default: None, means using ToyNet")
    parser.add_argument("--save", metavar="SAVE", type=str, default='../assets/mnist_cnn.onnx',
                        help="Saving onnx path, default: '../assets/mnist_cnn.onnx'")
    parser.add_argument("--dynamic", action="store_true", default=False,
                        help="Is the batch dimension dynamically set, default: False")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def load_model(pytorch_model=None):
    if pytorch_model is None:
        model = ToyNet()
    else:
        model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))
        assert pytorch_model in model_names

        model = models.__dict__[pytorch_model](pretrained=True)

    model.eval()
    return model


def check_onnx(onnx_path='pytorch.onnx'):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def check_output(x, torch_out, onnx_path='pytorch.onnx'):
    # ValueError: This ORT build has ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] enabled. \
    # Since ORT 1.9, you are required to explicitly set the providers parameter when instantiating InferenceSession.
    # For example, onnxruntime.InferenceSession(..., providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], ...)
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print("Onnx info:")
    print(f"    input: {ort_session.get_inputs()[0]}")
    print(f"    output: {ort_session.get_outputs()[0]}")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(x.shape, ort_outs[0].shape)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def export_to_onnx(torch_model, batch_size=1, img_dim=1, img_size=28, onnx_path="pytorch.onnx", is_dynamic=False):
    assert isinstance(torch_model, nn.Module)

    if is_dynamic:
        batch_size = 5

    # Input to the model
    x = torch.randn(batch_size, img_dim, img_size, img_size, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    if not is_dynamic:
        torch.onnx.export(torch_model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          onnx_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          )
    else:
        torch.onnx.export(torch_model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          onnx_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}}
                          )

    check_onnx(onnx_path=onnx_path)
    check_output(x, torch_out, onnx_path=onnx_path)


def main(args):
    model = load_model(pytorch_model=args.model)
    if args.model is None:
        batch_size = 1
        img_dim = 1
        img_size = 28
    else:
        batch_size = 1
        img_dim = 3
        img_size = 224

    onnx_path = args.save
    is_dynamic = args.dynamic
    export_to_onnx(model, batch_size=batch_size, img_dim=img_dim, img_size=img_size,
                   onnx_path=onnx_path, is_dynamic=is_dynamic)
    print(f"Save to {onnx_path}")


if __name__ == '__main__':
    args = parse_opt()
    main(args)
