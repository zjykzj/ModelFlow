# ModelFlow — Specification Index

> **Status:** Draft
> **Version:** 0.1

## Overview

ModelFlow 是一个计算机视觉模型部署工具集，专注于**模型推理、评估与导出**。支持多种视觉任务、多种推理后端和多种编程语言，以 **Pipeline** 模式统一各模块接口。

## Architecture at a Glance

```
ModelFlow/
├── modelflow/       # Python 实现：推理 + 评估 + 可视化
├── export/          # 模型导出（pt → onnx → tensorrt/triton）
├── cpp/             # C++ 推理（OpenCV + onnxruntime/tensorrt）
├── samples/         # 使用示例
├── assets/          # 测试资源
├── models/          # 模型文件
└── specs/           # 规格文档
    └── modules/     #    ㊀ 模块层 — 实现规格（HOW）
```

### 推理管线（Pipeline）模式

```
InferencePipeline = Preprocessor + Backend + Postprocessor
```

所有模块遵循同一语义，Python 与 C++ 实现精度对齐。

### 设计原则

| 原则 | 说明 |
|------|------|
| **模块独立** | `modelflow/`、`export/`、`cpp/` 三者平级，无相互依赖 |
| **Pipeline 驱动** | 每个推理任务 = 预处理 + 后端 + 后处理 |
| **PyTorch 仅用于导出** | Python 推理只用 ONNX Runtime / TensorRT / Triton |
| **评估/可视化依赖 DataFlow-CV** | 通过桥接调用 [DataFlow-CV](https://github.com/zjykzj/DataFlow-CV) 的 metrics 和 visualizer |
| **C++ 后端可提取** | ONNX Runtime 和 TensorRT 后端各有独立 CMakeLists.txt |
| **注册机制** | 新增后端/任务/数据集不改核心框架代码 |

## Specification Layers

| Layer | Directory | What it defines |
|-------|-----------|-----------------|
| **Modules** | [`modules/`](modules/index.md) | 实现规格（HOW）— 三个模块的架构、接口、扩展方式 |

## Documents

| # | Module | Document | Purpose |
|---|--------|----------|---------|
| 1 | Architecture | [`modules/spec_architecture.md`](modules/spec_architecture.md) | 三层独立模块设计、模块间关系、Pipeline 模式、数据流 |
| 2 | Python | [`modules/spec_python.md`](modules/spec_python.md) | `modelflow/` 包：抽象基类、推理后端、处理器、评估器、注册机制 |
| 3 | Export | [`modules/spec_export.md`](modules/spec_export.md) | 模型导出管线：pt → onnx → tensorrt/triton |
| 4 | C++ | [`modules/spec_cpp.md`](modules/spec_cpp.md) | C++ 推理模块：独立可提取后端、精度对齐 |

## Reading Order

| 读者 | 推荐顺序 |
|------|----------|
| **架构理解** | `modules/spec_architecture.md` → `modules/spec_python.md` |
| **Python 开发** | `modules/spec_python.md` → `modules/spec_architecture.md` |
| **模型导出** | `modules/spec_export.md` |
| **C++ 部署** | `modules/spec_cpp.md` |

## Task Coverage

| 任务 | Python | C++ | Export |
|------|--------|-----|--------|
| Classification | ✅ | ✅ | ✅ |
| Detection | ✅ | ✅ | ✅ |
| Instance Segmentation | ✅ | ✅ | ✅ |
| Semantic Segmentation | ✅ | ❌ | ✅ |
| Multi-modal (CLIP) | ✅ | ❌ | ✅ |
