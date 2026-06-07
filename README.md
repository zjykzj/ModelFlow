<div align="center"><a title="" href="git@github.com:zjykzj/ModelFlow.git"><img align="center" src="./assets/logos/ModelFlow.svg"></a></div>

<p align="center">
  Model Eval & Export & Infer — 计算机视觉模型部署工具集
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
</p>

## 简介

ModelFlow 是一个计算机视觉模型部署工具集，聚焦 **模型评估、导出与推理**。
支持多种视觉任务、多种推理后端，以 **Pipeline** 模式统一各模块接口。

```
InferencePipeline = Preprocessor + Backend + Postprocessor
```

### 核心特性

- **统一 Pipeline 模式**：预处理 + 推理后端 + 后处理 三段式解耦
- **多后端支持**：ONNX Runtime / TensorRT (FP16/INT8) / Triton Inference Server
- **模型导出管线**：PyTorch → ONNX → TensorRT → Triton 全链路
- **评估框架**：分类/检测/分割评估，DataFlow-CV 桥接 mAP
- **注册机制**：新增后端/任务/数据集无需改核心框架

## 架构

```
modelflow/           # Python 推理/评估/可视化核心包
  ├── core/          # 抽象基类、注册机制、类型枚举
  ├── backends/      # ONNX / TensorRT / Triton 推理后端
  ├── processors/    # 前/后处理器（classify/detect/segment/multimodal）
  ├── pipelines/     # Pipeline 工厂函数
  ├── datasets/      # 数据集加载器
  ├── evaluators/    # 评估编排器（DataFlow-CV 桥接）
  ├── metrics/       # 本地指标实现
  └── viz/           # 可视化

export/              # 模型导出（pt → onnx → tensorrt → triton）
  ├── onnx/          # torchvision + Ultralytics ONNX 导出
  ├── tensorrt/      # FP16/INT8 引擎构建（PyTorch/PyCUDA 校准器）
  ├── triton/        # Triton 配置生成 + 仓库管理
  └── scripts/       # 校准数据生成

specs/               # 规格文档
  ├── modules/       # 模块架构设计
  └── export/        # ONNX/TensorRT/Triton 知识层
```

详细架构文档见 [`specs/`](specs/index.md)。

## 快速开始

### 安装

```bash
# 基础依赖
pip install torch torchvision onnx onnxruntime

# 可选后端
pip install tensorrt           # TensorRT
pip install tritonclient[grpc] # Triton
pip install pycuda             # INT8 PyCUDA 校准器
```

### 分类推理

```python
from modelflow.pipelines import create_classify_pipeline

pipeline = create_classify_pipeline(
    model_path="efficientnet_b0.onnx",
    class_list=["cat", "dog", "bird"],
    backend="onnxruntime",
)
result = pipeline(image)
print(result["class_ids"], result["scores"])  # top-5
```

### 检测推理

```python
from modelflow.pipelines import create_detect_pipeline
from modelflow.cfgs.coco import class_list

pipeline = create_detect_pipeline(
    model_path="yolov8s.onnx",
    class_list=class_list,
    backend="onnxruntime",
)
result = pipeline(image, conf_thres=0.25, iou_thres=0.45)
# {"boxes": ndarray(N,4), "scores": ndarray(N,), "class_ids": ndarray(N,)}
```

### 模型导出

```bash
# L1: PT → ONNX
python3 -m export.onnx.convert --model efficientnet_b0 --save model.onnx
python3 -m export.onnx.ultralytics yolov8s --save yolov8s.onnx

# L2: ONNX → TensorRT FP16
python3 -m export.tensorrt.build_fp16 --onnx model.onnx --save model_fp16.engine

# L3: + Triton 配置
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt --task detect --save ./models/triton/
```

### 评估

```python
from modelflow.evaluators import DetectEvaluator
from modelflow.datasets import COCODetectionDataset
from modelflow.cfgs.coco import class_list

dataset = COCODetectionDataset("val2017/", class_list, anno_json="annotations.json")
evaluator = DetectEvaluator(pipeline, dataset, gt_json="annotations.json")
metrics = evaluator.run()
# {"mAP": 0.437, "AP50": 0.602, ...}
```

详细使用见 [`modelflow/README.md`](modelflow/README.md)。

## 支持的任务与模型

| 任务 | Preprocessor | Backend | Postprocessor |
|------|-------------|---------|---------------|
| Classification | Resize+Crop+Normalize | ONNX/TRT/Triton | softmax top-5 |
| Detection | LetterBox → /255 | ONNX/TRT/Triton | NMS + coordinate scaling |
| Instance Segmentation | LetterBox → /255 | ONNX/TRT/Triton | NMS + proto mask |
| Semantic Segmentation | Resize+Normalize | ONNX/TRT/Triton | argmax + colormap |
| Multi-modal (CLIP) | CLIP standard | ONNX/TRT/Triton | similarity ranking |

### 模型来源

| 来源 | 任务 | 模型示例 |
|------|------|---------|
| torchvision | Classification | ResNet, EfficientNet, MobileNet, ViT, ConvNeXt |
| Ultralytics | Detection | YOLOv5, YOLOv8, YOLO11 |
| Ultralytics | Segmentation | YOLOv8-seg |
| Ultralytics | Pose | YOLOv8-pose |
| OpenAI | Multi-modal | CLIP, OpenCLIP |

## 导出深度等级

| 等级 | 产出 | 推理方式 |
|------|------|---------|
| **L1** | `.onnx` | ONNX Runtime (CPU/GPU) |
| **L2** | `.onnx` + `.engine` | TensorRT GPU (FP16/INT8) |
| **L3** | `.onnx` + `.engine` + Triton config | Triton Server |

## 项目结构

```
ModelFlow/
├── modelflow/           # Python 推理/评估核心包
├── export/              # 模型导出管线
├── specs/               # 规格文档
├── eval/                # 评估基准入口
├── samples/             # 使用示例
├── llms/                # CLIP/OpenCLIP 评估
├── models/              # 预训练模型文件
└── assets/              # 测试资源
```

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/ModelFlow/issues) or submit PRs.

## License

[Apache License 2.0](LICENSE) © 2021 zjykzj
