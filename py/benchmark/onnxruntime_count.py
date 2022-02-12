# -*- coding: utf-8 -*-

"""
@date: 2021/7/24 下午1:54
@file: onnxruntime_count.py
@author: zj
@description: 
"""

import os
import argparse
import time
import torch
import onnxruntime

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_size', type=int, default=224, help='Data Size. Default: 224')
    parser.add_argument('onnx_name', type=str, default=None, help='ONNX Path')

    args = parser.parse_args()
    # print(args)
    return args


def get_onnx_model(onnx_name):
    ort_session = onnxruntime.InferenceSession(onnx_name)
    return ort_session


def compute_model_time(ort_session, data_shape, num=100):
    # compute ONNX Runtime output prediction
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    t1 = 0.0
    begin = time.time()
    for i in tqdm(range(num)):
        time.sleep(0.01)
        data = torch.randn(data_shape).numpy()
        ort_inputs = {input_name: data}

        start = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        if i > num // 2:
            t1 += time.time() - start
    t2 = time.time() - begin
    print(f'one process need {t2 / num:.3f}s, model compute need: {t1 / (num // 2):.3f}s')


if __name__ == '__main__':
    args = parse_args()

    data_size = int(args.data_size)
    onnx_name = os.path.abspath(args.onnx_name)

    model = get_onnx_model(onnx_name)

    compute_model_time(model, (1, 3, data_size, data_size))
