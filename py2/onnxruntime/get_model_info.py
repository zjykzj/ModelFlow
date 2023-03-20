# -*- coding: utf-8 -*-

"""
@date: 2023/3/14 上午10:34
@file: get_input_output_info.py
@author: zj
@description: 
"""

import onnx

onnx_model = onnx.load("../../assets/mnist_cnn.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime as ort
import numpy as np



ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})



1. 加载模型
    模型初始化
    推理平台指定
3. 加载数据
    4. 数据初始化
5. 获取输入节点、输出节点信息
6. 打印输出结果
7. 推理（warmup + 平均推理时间）
    单张图像分类推理
    批量图像分类推理
    单张图像检测推理
    批量图像检测推理