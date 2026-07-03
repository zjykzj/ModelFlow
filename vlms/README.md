# VLMs — Vision-Language Models

视觉-语言多模态模型模块。本模块包含需要**文本输入**的多模态模型前/后处理器和评估脚本，区别于 `modelflow/` 中单视觉输入的纯图像推理管线。

## 模块定位

```
vlms/                        # 多模态：图像 + 文本双输入
├── contrastive/             # 对比学习 / Embedding 类
├── detection/               # 开放词汇检测类
├── generative/              # 生成式 VLM 类
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

## 三类 VLM 实现逻辑对比

VLM 虽然都是"图像 + 文本"双输入，但底层推理范式完全不同，分为三个独立的类别：

| 维度 | 对比 / Embedding | 开放词汇检测 | 生成式 VLM |
|------|------------------|--------------|------------|
| **代表模型** | CLIP, OpenCLIP | YOLOE, YOLO-WORLD, Grounding DINO | Qwen-VL, InternVL, LLaVA |
| **核心思想** | 双塔双编码器，对比学习对齐图文特征 | 检测器 + 文本编码器替代固定类别头 | 图像 tokenize 后拼入 LLM，自回归生成 |
| **推理流程** | 图像编码 → 文本编码 → cosine similarity | 图像特征 → 文本特征 → 检测头 → NMS | 图像 tokenize → concat text tokens → LLM autoregressive decode |
| **输出** | 相似度分数 / 分类 logits | bbox + 类别标签 | 自然语言文本 |
| **后处理** | softmax / top-k | NMS + box decode | token detokenization |
| **特有组件** | — | — | tokenizer, KV cache, chat template, streaming |
| **典型用途** | Zero-shot 分类、图文检索、特征提取 | 开放词汇目标检测 | 视觉问答、图像描述、多轮对话 |
| **推理模式** | 前向一次，无状态 | 前向一次，无状态 | 自回归逐 token 生成，有状态 (KV cache) |

### 对比 / Embedding 类

```text
图像 ──► ImageEncoder ──► image_features ──┐
                                            ├──► cosine similarity ──► logits
文本 ──► TextEncoder  ──► text_features  ──┘
```

- **双塔结构**：图像和文本分别编码，仅通过对比损失在训练时交互
- **推理时两路独立**：可预先计算文本特征缓存（如 prompt ensemble），推理只跑图像编码器
- **后处理极简**：softmax / top-k，无 NMS、无 token 生成
- **无状态**：每次推理独立，无 KV cache

### 开放词汇检测类

```text
图像 ──► ImageEncoder ──► image_features ──┐
                                            ├──► DetectionHead ──► NMS ──► bboxes + labels
文本 ──► TextEncoder  ──► text_features  ──┘
```

- **检测器架构**：本质是目标检测器，文本 prompt 替代了固定的分类头权重
- **文本作为条件**：文本特征通过 cross-attention 或 prompt fusion 注入检测管线
- **后处理是检测管线**：NMS + bbox decode，输出结构化检测结果
- **与 CLIP 的本质区别**：CLIP 输出的是"这张图是不是猫"（全图级别），检测输出的是"猫在哪里"（实例级别 + 定位）

### 生成式 VLM 类

```text
图像 ──► VisionEncoder ──► visual tokens ──┐
                                            ├──► LLM ──► autoregressive decode ──► text
文本 ──► Tokenizer     ──► text tokens  ──┘
```

- **LLM 为核心**：视觉编码器输出被投影为 LLM 可理解的 token 序列，拼接到文本 token 序列中
- **自回归生成**：逐 token 预测下一个 token，输出自然语言文本
- **全链路组件**：tokenizer → chat template → vision encoder → projector → LLM → sampler → detokenizer
- **有状态推理**：KV cache 跨 token 持久化，batch 推理需处理不等长序列
- **与前两类的本质区别**：前两类是判别式（discriminative），生成式是...生成式（generative）

## 现有算法

### CLIP (OpenAI) — 对比 / Embedding

- **论文**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)
- **核心思想**: 双塔结构 —— 图像编码器 (ViT/ResNet) + 文本编码器 (Transformer)，通过对比学习在 4 亿图文对上联合训练
- **能力**: Zero-shot 分类、图文检索、特征提取
- **本模块实现**: `CLIPImagePreprocessor` / `CLIPTextPreprocessor` / `CLIPPostprocessor`
- **评估**: CIFAR-10/100 上的 Zero-Shot 和 Linear Probe

### OpenCLIP — 对比 / Embedding

- **论文**: [Reproducible Scaling Laws for Contrastive Vision-Language Learning](https://arxiv.org/abs/2212.07143) (Cherti et al., 2022)
- **核心思想**: CLIP 的开源复现，使用 LAION 数据集，提供多种规模的 ViT backbone
- **本模块实现**: 基于 `open_clip_torch` 的评估样例
- **评估**: CIFAR-10/100 上的 Zero-Shot 和 Linear Probe（含 LR / KNN 分类头对比）

## 待实现算法

| 算法 | 类别 | 论文 | 核心能力 | 状态 |
|------|------|------|----------|------|
| **YOLOE** | 检测 | [YOLOE: Real-Time Seeing Anything](https://arxiv.org/abs/2504.02793) (2025) | 开放词汇目标检测，文本提示 | 计划中 |
| **YOLO-WORLD** | 检测 | [YOLO-World: Real-Time Open-Vocabulary Object Detection](https://arxiv.org/abs/2401.17270) (2024) | 开放词汇实时目标检测 | 计划中 |
| **Grounding DINO** | 检测 | [Grounding DINO: Marrying DINO with Grounded Pre-Training](https://arxiv.org/abs/2303.05499) (2023) | 开放词汇目标检测 | 候选 |
| **Qwen-VL** | 生成式 | [Qwen-VL: A Versatile Vision-Language Model](https://arxiv.org/abs/2308.12966) (2023) | 视觉问答、图像描述、多轮对话 | 计划中 |
| **InternVL** | 生成式 | [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2312.14238) (2023) | 视觉问答、文档理解 | 候选 |

## 目录结构

```
vlms/
├── README.md                 # 本文件
├── __init__.py               # 包初始化
│
├── clip/                     # OpenAI CLIP（对比/Embedding）
│   ├── __init__.py           # CLIPImagePreprocessor, CLIPTextPreprocessor, CLIPPostprocessor
│   ├── preprocess.py         # 图像/文本预处理
│   ├── postprocess.py        # 后处理（logit → probability）
│   ├── samples/              # 评估样例
│   │   ├── README.md
│   │   ├── clip_zero_shot_cifar.py
│   │   └── clip_linear_probe_cifar.py
│   └── tests/
│       └── test_processors.py
│
└── openclip/                 # OpenCLIP（对比/Embedding）
    └── samples/              # 评估样例
        ├── README.md
        ├── results.md
        ├── openclip_zero_shot_cifar.py
        └── openclip_linear_probe_cifar.py
```

## 开发规范

1. **前后端分离**: 每个算法提供独立的 `Preprocessor`（图像 + 文本）和 `Postprocessor`，遵循 `modelflow/` 的 Pipeline 模式
2. **按类别组织**: 新增算法按其推理范式归入对应类别（contrastive / detection / generative），类别内部保持一致的接口约定
3. **生成式 VLM 额外注意**: tokenizer、chat template、KV cache 管理、streaming 输出等组件应模块化，便于跨模型复用
4. **评估脚本**: 放在 `samples/` 目录，可独立运行，不依赖 `modelflow/`
5. **测试**: 放在 `tests/` 目录，使用 pytest，无外部模型依赖
