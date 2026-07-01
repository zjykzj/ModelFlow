# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/8 18:24
@File    : openclip_linear_probe_cifar.py
@Author  : zj
@Description:

python openclip_linear_probe_cifar.py --dataset cifar10 --classifier logistic

python openclip_linear_probe_cifar.py --dataset cifar10 --classifier knn --k 1

python openclip_linear_probe_cifar.py --dataset cifar100 --classifier knn --k 5

"""

import os
import argparse
import torch
import open_clip
import numpy as np
from tqdm import tqdm

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


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
    """从 dataloader 中提取所有图像的 OpenCLIP 视觉特征（L2 归一化）"""
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
    parser = argparse.ArgumentParser(description="Linear Probe evaluation of OpenCLIP on CIFAR datasets.")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"],
                        help="Dataset to evaluate on (default: cifar10)")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="OpenCLIP model architecture (default: ViT-B-32)")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32",
                        help="Pretrained checkpoint name (default: laion400m_e32)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction (default: 64)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading (default: 4)")

    # 新增：分类器类型和 KNN 参数
    parser.add_argument("--classifier", type=str, default="logistic",
                        choices=["logistic", "knn"],
                        help="Classifier head: 'logistic' or 'knn' (default: logistic)")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of neighbors for KNN (used only when --classifier=knn, default: 1)")

    args = parser.parse_args()

    # ----------------------------
    # 1. 设置设备与模型
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print(f"🧠 Loading OpenCLIP model: {args.model} | Pretrained: {args.pretrained} ...")

    model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained=args.pretrained
    )
    model.to(device).eval()
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
    # 4. 训练分类器（Logistic 或 KNN）
    # ----------------------------
    if args.classifier == "logistic":
        print("🛠️ Training Logistic Regression classifier...")
        classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        classifier.fit(train_features, train_labels)
        print("✅ Logistic classifier training completed.")
    elif args.classifier == "knn":
        print(f"🛠️ Setting up KNN classifier with k={args.k} (metric=cosine, since features are normalized)...")
        # 注意：因为特征已 L2 归一化，余弦相似度 = 内积 = -欧氏距离平方（单调关系）
        # 所以可以用 'cosine' 距离，或 'euclidean'（效果等价，但 cosine 更直观）
        classifier = KNeighborsClassifier(
            n_neighbors=args.k,
            metric='cosine',  # 支持 cosine
            algorithm='brute',  # 对于 <100K 样本，brute 足够快且精确
            n_jobs=-1
        )
        classifier.fit(train_features, train_labels)
        print("✅ KNN classifier ready (no training needed).")

    # ----------------------------
    # 5. 评估测试集准确率
    # ----------------------------
    print("🧪 Evaluating on test set...")
    test_preds = classifier.predict(test_features)
    accuracy = (test_preds == test_labels).mean()

    print("\n" + "=" * 70)
    print(f"🎯 OpenCLIP Frozen Feature Classification Results")
    print(f"   Model:       {args.model}")
    print(f"   Pretrained:  {args.pretrained}")
    print(f"   Dataset:     {args.dataset.upper()}")
    print(f"   Classifier:  {args.classifier}")
    if args.classifier == "knn":
        print(f"   k (top-N):   {args.k}")
    print(f"   Train Size:  {len(train_labels)}")
    print(f"   Test Size:   {len(test_labels)}")
    print(f"   Accuracy:    {accuracy:.4%}  ({int(accuracy * len(test_labels))}/{len(test_labels)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
