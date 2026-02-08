# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/8 18:24
@File    : openclip_linear_probe_cifar.py
@Author  : zj
@Description:

root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python3 openclip_linear_probe_cifar.py
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 | Pretrained: laion400m_e32 ...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and frozen.
📂 Loading CIFAR10 training set...
📂 Loading CIFAR10 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:24<00:00, 31.55it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 30.37it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Training Logistic Regression classifier...
✅ Classifier training completed.
🧪 Evaluating on test set...

============================================================
🎯 OpenCLIP Linear Probe Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR10
   Train Size:  50000
   Test Size:   10000
   Accuracy:    94.7500%  (9475/10000)
============================================================
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python3 openclip_linear_probe_cifar.py --dataset cifar100
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 | Pretrained: laion400m_e32 ...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and frozen.
📂 Loading CIFAR100 training set...
📂 Loading CIFAR100 test set...
🔍 Extracting features from training set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:25<00:00, 31.10it/s]
🔍 Extracting features from test set...
Extracting features: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 28.98it/s]
📊 Feature shape - Train: (50000, 512), Test: (10000, 512)
🛠️ Training Logistic Regression classifier...
✅ Classifier training completed.
🧪 Evaluating on test set...

============================================================
🎯 OpenCLIP Linear Probe Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR100
   Train Size:  50000
   Test Size:   10000
   Accuracy:    78.7900%  (7879/10000)
============================================================

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
    args = parser.parse_args()

    # ----------------------------
    # 1. 设置设备与模型
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print(f"🧠 Loading OpenCLIP model: {args.model} | Pretrained: {args.pretrained} ...")

    # 加载模型、transform（preprocess）——注意：不需要 tokenizer
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
    # 4. 训练线性分类器（Logistic Regression）
    # ----------------------------
    print("🛠️ Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0
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
    print(f"🎯 OpenCLIP Linear Probe Classification Results")
    print(f"   Model:       {args.model}")
    print(f"   Pretrained:  {args.pretrained}")
    print(f"   Dataset:     {args.dataset.upper()}")
    print(f"   Train Size:  {len(train_labels)}")
    print(f"   Test Size:   {len(test_labels)}")
    print(f"   Accuracy:    {accuracy:.4%}  ({int(accuracy * len(test_labels))}/{len(test_labels)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
