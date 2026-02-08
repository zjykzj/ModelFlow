# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/8 17:16
@File    : eval_cifar.py
@Author  : zj
@Description: 
"""

import os
import torch
import clip

from tqdm import tqdm

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# ----------------------------
# 1. 设置设备与模型
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ----------------------------
# 2. CIFAR-10 类别名称（必须按官方顺序）
# ----------------------------
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ----------------------------
# 3. 构造文本 prompts 并编码（只做一次）
# ----------------------------
with torch.no_grad():
    text_inputs = clip.tokenize([f"a photo of a {c}" for c in cifar10_classes]).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # L2 归一化

# ----------------------------
# 4. 加载 CIFAR-10 测试集（使用 CLIP 的 preprocess）
# ----------------------------
# 注意：CIFAR10 返回的是 PIL Image，所以可以直接用 preprocess
test_dataset = CIFAR10(root=os.path.expanduser("~/.cache"), train=False, download=True, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# ----------------------------
# 5. 批量推理与评估
# ----------------------------
correct = 0
total = 0
logit_scale = model.logit_scale.exp()

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 编码图像
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # 计算 logits: [B, K]
        logits = logit_scale * (image_features @ text_features.T)  # shape: (batch_size, 10)

        # 预测类别
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\n✅ CLIP ViT-B/32 Zero-Shot Accuracy on CIFAR-10: {accuracy:.4f} ({correct}/{total})")
