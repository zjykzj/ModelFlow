# -*- coding: utf-8 -*-

"""
@date: 2021/7/24 下午1:19
@file: zcls_pytorch_to_onnx.py
@author: zj
@description: convert zcls pytorch into onnx format
"""

import os
import torch
import argparse

from functools import reduce

from zcls.config import cfg
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.recognizers.build import build_recognizer

from misc import check_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_size', type=int, default=224, help='Data Size. Default: 224')
    parser.add_argument('zcls_config_path', type=str, default=None, help='ZCls Config Path')
    parser.add_argument('res_onnx_name', type=str, default=None, help='Result ONNX Name')

    args = parser.parse_args()
    # print(args)
    return args


def get_zcls_model(cfg, device):
    model = build_recognizer(cfg, device=device)
    model.eval()
    return model


def export_to_onnx(model, onnx_name, data_shape, device):
    data_length = reduce(lambda x, y: x * y, data_shape)
    data = torch.arange(data_length).float().reshape(data_shape).to(device)

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


if __name__ == '__main__':
    args = parse_args()

    data_size = int(args.data_size)
    zcls_config_path = os.path.abspath(args.zcls_config_path)
    res_onnx_name = os.path.abspath(args.res_onnx_name)

    device = torch.device('cpu')
    cfg.merge_from_file(zcls_config_path)
    model = get_zcls_model(cfg, device)

    data, torch_out = export_to_onnx(model, res_onnx_name, (1, 3, data_size, data_size), device)
    check_model(onnx_name=res_onnx_name, verbose=True)
