# -*- coding: utf-8 -*-

"""
@date: 2024/5/1 下午4:18
@file: classify.py
@author: zj
@description: 
"""

import json
import cv2
import numpy as np
from PIL import Image

# 加载分类模型
net = cv2.dnn.readNet("../../export/resnet50_pytorch.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# # 加载 ImageNet 类别列表
# with open('../../assets/imagenet/imagenet.names', 'r') as f:
#     class_labels = [line.strip() for line in f.readlines()]
# 加载 ImageNet 类别列表
imagenet_path = '../../assets/imagenet/imagenet_class_index.json'
with open(imagenet_path, 'r') as f:
    data_dict = json.load(f)

# 加载图像并进行预处理
image_np = cv2.imread("../../assets/imagenet/n02113023/ILSVRC2012_val_00010244.JPEG")
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)


# image_pil = Image.open("../../assets/imagenet/n02113023/ILSVRC2012_val_00010244.JPEG")
# image_np = np.array(image_pil)

# 定义预处理步骤
def preprocess_image(image_np):
    # 获取图像的尺寸
    h, w = image_np.shape[:2]
    # 计算缩放比例
    if h < w:
        new_h, new_w = 256, int(w * 256 / h)
    else:
        new_h, new_w = int(h * 256 / w), 256
    # 缩放图像
    resized_image = cv2.resize(image_np, (new_w, new_h))

    # 计算中心裁剪区域
    top = (new_h - 224) // 2
    left = (new_w - 224) // 2
    bottom = top + 224
    right = left + 224
    # 中心裁剪图像
    cropped_image = resized_image[top:bottom, left:right]
    print(f"cropped_image.shape: {cropped_image.shape} - dtype: {cropped_image.dtype}")

    # 定义均值和标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalized_image = cv2.dnn.blobFromImage(cropped_image, 1 / 255.0, (224, 224), mean * 255.0)
    print(f"normalized_image.shape: {normalized_image.shape} - dtype: {normalized_image.dtype}")
    # 标准差归一化
    for i in range(3):
        normalized_image[0, i] /= std[i]

    # cropped_image = cropped_image / 255.0
    # # 假设 image 是经过处理的图像数据，形状为 (H, W, C)
    # # 这里假设 image 是一个三维的 NumPy 数组，表示图像的像素值
    # # 对图像进行均值归一化处理
    # normalized_image = (cropped_image - mean) / std
    # # 将图像数据转换为格式为 (N, C, H, W) 的 4 维数组
    # # 这里假设只有一张图像，因此 N=1
    # normalized_image = normalized_image.transpose((2, 0, 1))  # 将通道维度移到第二个维度
    # normalized_image = np.expand_dims(normalized_image, axis=0)  # 增加一个维度作为批处理维度
    print(f"normalized_image.shape: {normalized_image.shape} - dtype: {normalized_image.dtype}")

    return normalized_image


# 定义预处理步骤
def preprocess_image2(image_np):
    # 获取图像的尺寸
    h, w = image_np.shape[:2]
    # 计算缩放比例
    if h < w:
        new_h, new_w = 256, int(w * 256 / h)
    else:
        new_h, new_w = int(h * 256 / w), 256
    # 缩放图像
    resized_image = cv2.resize(image_np, (new_w, new_h))

    # 计算中心裁剪区域
    top = (new_h - 224) // 2
    left = (new_w - 224) // 2
    bottom = top + 224
    right = left + 224
    # 中心裁剪图像
    cropped_image = resized_image[top:bottom, left:right]
    print(f"cropped_image.shape: {cropped_image.shape} - dtype: {cropped_image.dtype}")

    # 定义均值和标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    cropped_image = cropped_image / 255.0
    # 假设 image 是经过处理的图像数据，形状为 (H, W, C)
    # 这里假设 image 是一个三维的 NumPy 数组，表示图像的像素值
    # 对图像进行均值归一化处理
    normalized_image = (cropped_image - mean) / std
    # 将图像数据转换为格式为 (N, C, H, W) 的 4 维数组
    # 这里假设只有一张图像，因此 N=1
    normalized_image = normalized_image.transpose((2, 0, 1))  # 将通道维度移到第二个维度
    normalized_image = np.expand_dims(normalized_image, axis=0)  # 增加一个维度作为批处理维度
    print(f"normalized_image.shape: {normalized_image.shape} - dtype: {normalized_image.dtype}")

    return normalized_image


normalized_image = preprocess_image(image_np)

# 设置输入数据
net.setInput(normalized_image)
# 执行前向传播
outputs = net.forward()
print(outputs.shape)
print(outputs[0, :10])

# 计算 softmax
probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)

# 获取前五位的结果
top5_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :5]
top5_prob = np.take_along_axis(probabilities, top5_indices, axis=1)

# 输出每个样本的前五位结果
for i in range(outputs.shape[0]):  # 遍历每个样本
    print(f"Sample {i + 1}:")
    for j in range(5):  # 输出前五位结果
        idx = top5_indices[i, j]
        # label = class_labels[idx]
        label = data_dict[str(idx)][1]
        probability = top5_prob[i, j]
        print(f"Top {j + 1}: Class index: {idx}, Class label: {label}, Probability: {probability}")
