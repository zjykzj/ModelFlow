# -*- coding: utf-8 -*-

"""
@date: 2021/7/24 下午1:53
@file: compute_zcls_pytorch.py
@author: zj
@description: Calculate the model size, flops and reasoning time
"""

import os
import argparse
import time
import torch
import warnings

from tqdm import tqdm
from thop import profile

from zcls.config import cfg
from zcls.model.recognizers.build import build_recognizer

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_size', type=int, default=224, help='Data Size. Default: 224')
    parser.add_argument('zcls_config_path', type=str, default=None, help='ZCls Config Path')

    args = parser.parse_args()
    # print(args)
    return args


def get_zcls_model(cfg, device):
    model = build_recognizer(cfg, device=device)
    model.eval()
    return model


def computer_flops_and_params(model, data_shape=(1, 3, 224, 224), device=torch.device('cpu')):
    input = torch.randn(data_shape).to(device)
    flops, params = profile(model, inputs=(input,), verbose=False)

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))


def compute_model_time(model, data_shape, device, num=100):
    model = model.to(device)
    model.eval()

    t1 = 0.0
    begin = time.time()
    for i in tqdm(range(num)):
        time.sleep(0.01)
        data = torch.randn(data_shape)
        start = time.time()
        model(data.to(device=device, non_blocking=True))
        if i > num // 2:
            t1 += time.time() - start
    t2 = time.time() - begin
    print(f'one process need {t2 / num:.3f}s, model compute need: {t1 / (num // 2):.3f}s')


if __name__ == '__main__':
    args = parse_args()

    data_size = int(args.data_size)
    zcls_config_path = os.path.abspath(args.zcls_config_path)

    device = torch.device('cpu')
    cfg.merge_from_file(zcls_config_path)
    model = get_zcls_model(cfg, device)

    computer_flops_and_params(model, (1, 3, data_size, data_size), device)
    compute_model_time(model, (1, 3, data_size, data_size), device)
