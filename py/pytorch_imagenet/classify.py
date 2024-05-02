# -*- coding: utf-8 -*-

"""
@date: 2024/5/1 下午3:56
@file: classify2.py
@author: zj
@description: 
"""

import json
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.models as models

# 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()

# 加载 ImageNet 类别列表
imagenet_path = '../../assets/imagenet/imagenet_class_index.json'
with open(imagenet_path, 'r') as f:
    data_dict = json.load(f)

# 定义预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像并进行预处理
image = Image.open("../../assets/imagenet/n02113023/ILSVRC2012_val_00010244.JPEG")
image_tensor = preprocess(image)
image_tensor = image_tensor.unsqueeze(0)  # 添加 batch 维度

# 执行推理
with torch.no_grad():
    outputs = model(image_tensor)
    print(outputs[0, :10])

# 计算分类概率
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# 获取前五位的结果
top5_prob, top5_indices = torch.topk(probabilities, 5)

# 输出前五位的结果
for i in range(5):
    idx = top5_indices[i].item()
    label = data_dict[str(idx)][1]
    probability = top5_prob[i].item()
    print(f"Top {i + 1}: Class index: {idx}, Class label: {label}, Probability: {probability}")
