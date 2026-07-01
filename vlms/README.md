# VLMs — Vision-Language Models

视觉-语言多模态模型模块。本模块包含需要**文本输入**的多模态模型前/后处理器和评估脚本，区别于 `modelflow/` 中单视觉输入的纯图像推理管线。

## 模块定位

```
vlms/                        # 多模态：图像 + 文本双输入
├── clip/                    # OpenAI CLIP
├── openclip/                # OpenCLIP
├── yoloe/                   # YOLOE (待实现)
├── yolo-world/              # YOLO-WORLD (待实现)
└── ...

modelflow/                   # 单模态：纯图像输入
├── processors/
│   ├── classify/            # 分类
│   ├── detect/              # 目标检测
│   ├── segment/             # 实例分割
│   ├── semantic_seg/        # 语义分割
│   └── ...
└── ...
```

## 现有算法

### CLIP (OpenAI)

- **论文**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- **核心思想**: 双塔结构 —— 图像编码器 (ViT/ResNet) + 文本编码器 (Transformer)，通过对比学习在 4 亿图文对上联合训练
- **能力**: Zero-shot 分类、图文检索、特征提取
- **本模块实现**: `CLIPImagePreprocessor` / `CLIPTextPreprocessor` / `CLIPPostprocessor`
- **评估**: CIFAR-10/100 上的 Zero-Shot 和 Linear Probe

### OpenCLIP

- **论文**: [Reproducible Scaling Laws for Contrastive Vision-Language Learning](https://arxiv.org/abs/2212.07143) (Cherti et al., 2022)
- **核心思想**: CLIP 的开源复现，使用 LAION 数据集，提供多种规模的 ViT backbone
- **本模块实现**: 基于 `open_clip_torch` 的评估样例
- **评估**: CIFAR-10/100 上的 Zero-Shot 和 Linear Probe（含 LR / KNN 分类头对比）

## 待实现算法

| 算法 | 论文 | 核心能力 | 状态 |
|------|------|----------|------|
| **YOLOE** | [YOLOE: Real-Time Seeing Anything](https://arxiv.org/abs/2504.02793) (2025) | 开放词汇目标检测，文本提示 | 计划中 |
| **YOLO-WORLD** | [YOLO-World: Real-Time Open-Vocabulary Object Detection](https://arxiv.org/abs/2401.17270) (2024) | 开放词汇实时目标检测 | 计划中 |
| **Grounding DINO** | [Grounding DINO: Marrying DINO with Grounded Pre-Training](https://arxiv.org/abs/2303.05499) (2023) | 开放词汇目标检测 | 候选 |
| **SAM + CLIP** | — | 分割 + 语义理解组合 | 候选 |

## 目录结构

```
vlms/
├── README.md                 # 本文件
├── __init__.py               # 包初始化
├── clip/                     # OpenAI CLIP
│   ├── __init__.py           # CLIPImagePreprocessor, CLIPTextPreprocessor, CLIPPostprocessor
│   ├── preprocess.py         # 图像/文本预处理
│   ├── postprocess.py        # 后处理（logit → probability）
│   ├── samples/              # 评估样例
│   │   ├── README.md
│   │   ├── clip_zero_shot_cifar.py
│   │   └── clip_linear_probe_cifar.py
│   └── tests/
│       └── test_processors.py
└── openclip/                 # OpenCLIP
    └── samples/              # 评估样例
        ├── README.md
        ├── results.md
        ├── openclip_zero_shot_cifar.py
        └── openclip_linear_probe_cifar.py
```

## 开发规范

1. **前后端分离**: 每个算法提供独立的 `Preprocessor`（图像 + 文本）和 `Postprocessor`，遵循 `modelflow/` 的 Pipeline 模式
2. **评估脚本**: 放在 `samples/` 目录，可独立运行，不依赖 `modelflow/`
3. **测试**: 放在 `tests/` 目录，使用 pytest，无外部模型依赖
