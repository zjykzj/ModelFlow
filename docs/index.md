# ModelFlow 文档中心

欢迎来到 ModelFlow 文档。ModelFlow 是一个计算机视觉模型部署工具集，覆盖**模型导出、推理、评估**全流程。

## 文档导航

### 推理与评估 — [`modelflow/`](../modelflow/README.md)

Python 推理/评估/可视化核心包，Pipeline = Preprocessor + Backend + Postprocessor 统一架构。

### 模型导出 — [`export/`](export/index.md)

独立、零外部依赖的模型导出管线，支持 **PyTorch → ONNX → TensorRT → Triton** 全链路。

| 文档 | 内容 |
|------|------|
| [导出总览与快速开始](export/index.md) | 整体架构、深度等级、快速入门 |
| [分类模型导出](export/classification.md) | ResNet / MobileNet / EfficientNet 等全链路导出 |
| [检测模型导出](export/detection.md) | YOLOv8 / YOLO11 / YOLO26 全链路导出 |
| [分割模型导出](export/segmentation.md) | YOLOv8-seg / YOLO11-seg / YOLO26-seg 全链路导出 |
| [TensorRT 深入指南](export/tensorrt.md) | FP16 / INT8 量化原理、校准器选型、性能调优 |
| [Triton 部署指南](export/triton.md) | config.pbtxt 配置、模型仓库、Docker 部署 |

### 规格文档 — [`specs/`](../specs/index.md)

模块架构设计和 ONNX/TensorRT/Triton 知识层，定义组件的接口契约和行为规范。

## 项目结构

```
ModelFlow/
├── docs/                # 📖 文档中心（当前目录）
│   └── export/          # 导出模块文档
├── modelflow/           # Python 推理/评估核心包
├── export/              # 模型导出管线
├── specs/               # 规格文档
├── samples/             # 使用示例
├── models/              # 预训练模型文件
├── tests/               # 测试
└── assets/              # 测试资源与图片
```

## 快速链接

- [安装依赖](#) — `pip install torch torchvision onnx onnxruntime`
- [运行测试](#) — `pytest tests/ -v`
- [SDD Agent 开发工作流](../specs/SDD_AGENT.md)
