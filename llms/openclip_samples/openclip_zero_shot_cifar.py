# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/8 18:12
@File    : openclip_zero_shot_cifar.py
@Author  : zj
@Description:

使用原始模板进行评估

root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python3 openclip_zero_shot_cifar.py
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 with pretrained weights: laion400m_e32...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and set to eval mode.
📊 Dataset size: 10000 samples | Classes: 10
🔤 Encoding text prompts...
✅ Encoded 10 text features.
🔍 Starting zero-shot evaluation...
Inference: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 29.55it/s]

============================================================
🎯 Zero-Shot Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR10
   Accuracy:    88.6600%  (8866/10000)
============================================================
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python3 openclip_zero_shot_cifar.py --dataset cifar100
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 with pretrained weights: laion400m_e32...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and set to eval mode.
📊 Dataset size: 10000 samples | Classes: 100
🔤 Encoding text prompts...
✅ Encoded 100 text features.
🔍 Starting zero-shot evaluation...
Inference: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 28.24it/s]

============================================================
🎯 Zero-Shot Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR100
   Accuracy:    67.0200%  (6702/10000)
============================================================

使用优化模板进行评估

root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python3 openclip_zero_shot_cifar.py
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 with pretrained weights: laion400m_e32...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and set to eval mode.
📊 Dataset size: 10000 samples | Classes: 10
🔤 Encoding text prompts...
✅ Encoded text features using 20 templates.
🔍 Starting zero-shot evaluation...
Inference: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 28.25it/s]

============================================================
🎯 Zero-Shot Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR10
   Accuracy:    88.8800%  (8888/10000)
============================================================
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples#
root@autodl-container-00e345b2a0-c853a801:~/zj/ModelFlow/llms/openclip_samples# python3 openclip_zero_shot_cifar.py --dataset cifar100
🚀 Using device: cuda
🧠 Loading OpenCLIP model: ViT-B-32 with pretrained weights: laion400m_e32...
/root/zj/open_clip/src/open_clip/factory.py:450: UserWarning: QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'laion400m_e32' (quick_gelu=True).
  warnings.warn(
✅ Model loaded and set to eval mode.
📊 Dataset size: 10000 samples | Classes: 100
🔤 Encoding text prompts...
✅ Encoded text features using 20 templates.
🔍 Starting zero-shot evaluation...
Inference: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:05<00:00, 29.46it/s]

============================================================
🎯 Zero-Shot Classification Results
   Model:       ViT-B-32
   Pretrained:  laion400m_e32
   Dataset:     CIFAR100
   Accuracy:    67.9000%  (6790/10000)
============================================================

"""

import os
import argparse
import torch
import open_clip
from tqdm import tqdm

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader


def get_dataset(dataset_name, preprocess):
    """根据名称返回对应的数据集和类别列表"""
    root = os.path.expanduser("~/.cache")
    if dataset_name.lower() == "cifar10":
        dataset = CIFAR10(root=root, train=False, download=True, transform=preprocess)
        classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
    elif dataset_name.lower() == "cifar100":
        dataset = CIFAR100(root=root, train=False, download=True, transform=preprocess)
        # CIFAR-100 官方细粒度类别（按顺序）
        classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'cifar10' or 'cifar100'.")
    return dataset, classes


def get_templates():
    """返回用于集成的手工模板列表"""
    return [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a large {}.',
        'a photo of a {} with sharp focus.',
        'a photo of many {}s.',
        'a close-up photo of a {}.',
        'a cropped photo of a {}.',
        'a bright photo of a {}.',
        'a dark photo of a {}.',
        'a photo of my {}.',
        'i love my {}!',
        'a plastic {}.',
        'a toy {}.',
        'a cartoon {}.'
    ]


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of OpenCLIP on CIFAR datasets.")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"],
                        help="Dataset to evaluate on (default: cifar10)")
    parser.add_argument("--model", type=str, default="ViT-B-32",  # 注意这里的模型名称可能不同
                        help="OpenCLIP model name (default: ViT-B-32)")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32",  # 添加预训练参数
                        help="Pretrained dataset for the model (default: laion400m_e32)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading (default: 4)")
    args = parser.parse_args()

    # ----------------------------
    # 1. 设置设备与模型
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print(f"🧠 Loading OpenCLIP model: {args.model} with pretrained weights: {args.pretrained}...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    print("✅ Model loaded and set to eval mode.")

    # ----------------------------
    # 2. 加载数据集与类别
    # ----------------------------
    test_dataset, class_names = get_dataset(args.dataset, preprocess)
    num_classes = len(class_names)
    print(f"📊 Dataset size: {len(test_dataset)} samples | Classes: {num_classes}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )

    # ----------------------------
    # 3. 编码文本 prompts（只做一次）
    # ----------------------------
    print("🔤 Encoding text prompts...")
    # with torch.no_grad():
    #     text_inputs = open_clip.tokenize([f"a photo of a {c}" for c in class_names]).to(device)
    #     text_features = model.encode_text(text_inputs)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    # print(f"✅ Encoded {num_classes} text features.")

    templates = get_templates()
    all_text_features = []

    with torch.no_grad():
        for template in templates:
            sentences = [template.format(c) for c in class_names]
            text_inputs = tokenizer(sentences).to(device)
            text_feats = model.encode_text(text_inputs)
            text_feats /= text_feats.norm(dim=-1, keepdim=True)
            all_text_features.append(text_feats)

        text_features = torch.stack(all_text_features).mean(dim=0)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    print(f"✅ Encoded text features using {len(templates)} templates.")

    # ----------------------------
    # 4. 批量推理与评估
    # ----------------------------
    correct = 0
    total = 0
    logit_scale = model.logit_scale.exp()

    print("🔍 Starting zero-shot evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * (image_features @ text_features.T)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print("\n" + "=" * 60)
    print(f"🎯 Zero-Shot Classification Results")
    print(f"   Model:       {args.model}")
    print(f"   Pretrained:  {args.pretrained}")
    print(f"   Dataset:     {args.dataset.upper()}")
    print(f"   Accuracy:    {accuracy:.4%}  ({correct}/{total})")
    print("=" * 60)


if __name__ == "__main__":
    main()
