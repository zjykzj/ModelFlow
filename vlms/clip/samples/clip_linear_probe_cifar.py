# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/8 17:50
@File    : clip_linear_probe.py
@Author  : zj
@Description:

root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/clip_samples# python3 clip_linear_probe_cifar.py
🚀 Using device: cuda
🧠 Loading CLIP model: ViT-B/32 ...
✅ Model loaded and frozen.
📂 Loading CIFAR10 training set...
📂 Loading CIFAR10 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:24<00:00, 31.72it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 30.74it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Training Logistic Regression classifier...
✅ Classifier training completed.
🧪 Evaluating on test set...

============================================================
🎯 Linear Probe Classification Results
   Model:       ViT-B/32
   Dataset:     CIFAR10
   Train Size:  50000
   Test Size:   10000
   Accuracy:    94.3200%  (9432/10000)
============================================================
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/clip_samples#
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/clip_samples# python3 clip_linear_probe_cifar.py --dataset cifar100
🚀 Using device: cuda
🧠 Loading CLIP model: ViT-B/32 ...
✅ Model loaded and frozen.
📂 Loading CIFAR100 training set...
📂 Loading CIFAR100 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:23<00:00, 32.83it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 31.36it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Training Logistic Regression classifier...
✅ Classifier training completed.
🧪 Evaluating on test set...

============================================================
🎯 Linear Probe Classification Results
   Model:       ViT-B/32
   Dataset:     CIFAR100
   Train Size:  50000
   Test Size:   10000
   Accuracy:    75.6200%  (7562/10000)
============================================================

"""

import os
import argparse
import torch
import clip
import numpy as np
from tqdm import tqdm

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression


def get_dataset(dataset_name, preprocess, train=True):
    """加载 CIFAR-10 或 CIFAR-100 数据集"""
    root = os.path.expanduser("~/.cache")
    if dataset_name.lower() == "cifar10":
        dataset = CIFAR10(root=root, train=train, download=True, transform=preprocess)
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        dataset = CIFAR100(root=root, train=train, download=True, transform=preprocess)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'cifar10' or 'cifar100'.")
    return dataset, num_classes


def extract_features(model, dataloader, device):
    """从 dataloader 中提取所有图像的 CLIP 特征（L2 归一化）"""
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 提取并归一化图像特征
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            all_features.append(image_features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_features), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Linear Probe evaluation of CLIP on CIFAR datasets.")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"],
                        help="Dataset to evaluate on (default: cifar10)")
    parser.add_argument("--model", type=str, default="ViT-B/32",
                        help="CLIP model name (default: ViT-B/32)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction (default: 64)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading (default: 4)")
    args = parser.parse_args()

    # ----------------------------
    # 1. 设置设备与模型
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print(f"🧠 Loading CLIP model: {args.model} ...")
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)  # 冻结整个模型
    print("✅ Model loaded and frozen.")

    # ----------------------------
    # 2. 加载训练集和测试集
    # ----------------------------
    print(f"📂 Loading {args.dataset.upper()} training set...")
    train_dataset, num_classes = get_dataset(args.dataset, preprocess, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    print(f"📂 Loading {args.dataset.upper()} test set...")
    test_dataset, _ = get_dataset(args.dataset, preprocess, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    # ----------------------------
    # 3. 提取训练集和测试集的特征
    # ----------------------------
    print("🔍 Extracting features from training set...")
    train_features, train_labels = extract_features(model, train_loader, device)

    print("🔍 Extracting features from test set...")
    test_features, test_labels = extract_features(model, test_loader, device)

    print(f"📊 Feature shape - Train: {train_features.shape}, Test: {test_features.shape}")

    # ----------------------------
    # 4. 训练线性分类器（Logistic Regression）
    # ----------------------------
    print("🛠️ Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0,  # 可尝试调整正则强度（如 C=0.1, 1.0, 10.0）
        # solver='lbfgs',  # 适用于中小规模多分类
        # multi_class='multinomial'
    )
    classifier.fit(train_features, train_labels)
    print("✅ Classifier training completed.")

    # ----------------------------
    # 5. 评估测试集准确率
    # ----------------------------
    print("🧪 Evaluating on test set...")
    test_preds = classifier.predict(test_features)
    accuracy = (test_preds == test_labels).mean()

    print("\n" + "=" * 60)
    print(f"🎯 Linear Probe Classification Results")
    print(f"   Model:       {args.model}")
    print(f"   Dataset:     {args.dataset.upper()}")
    print(f"   Train Size:  {len(train_labels)}")
    print(f"   Test Size:   {len(test_labels)}")
    print(f"   Accuracy:    {accuracy:.4%}  ({int(accuracy * len(test_labels))}/{len(test_labels)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
