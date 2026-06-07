# ModelFlow — Specification Index

> **Status:** Draft
> **Version:** 0.1

## Overview

ModelFlow 是一个计算机视觉模型部署工具集，专注于**模型推理、评估与导出**。以 **Pipeline** 模式（`InferencePipeline = Preprocessor + Backend + Postprocessor`）统一各模块接口，支持多种视觉任务、多种推理后端和多种编程语言。

## Specification Layers

| Layer | Directory | What it defines |
|-------|-----------|-----------------|
| **Modules** | [`modules/`](modules/index.md) | 实现规格（HOW）— 三个模块的架构、接口、扩展方式 |
| **Export Knowledge** | [`export/`](export/index.md) | 导出知识层（WHAT）— 格式原理、转换规范、选型依据 |

## Documents

| # | Module | Document | Purpose |
|---|--------|----------|---------|
| — | Development Guide | [`SSD_AGENT.md`](SSD_AGENT.md) | SSD Agent 开发方法论 — 如何基于 specs 进行开发 |
| 1 | Architecture | [`modules/spec_architecture.md`](modules/spec_architecture.md) | 三层独立模块设计、模块间关系、Pipeline 模式、数据流 |
| 2 | Python | [`modules/spec_python.md`](modules/spec_python.md) | `modelflow/` 包：抽象基类、推理后端、处理器、评估器、注册机制 |
| 3 | Export (架构) | [`modules/spec_export.md`](modules/spec_export.md) | 模型导出模块的架构：管线阶段、I/O 规范、精度验证标准 |
| 4 | Export (知识) | [`export/index.md`](export/index.md) | 导出知识库总览：路径路由、深度等级、模型支持矩阵 |
| 5 | Export (ONNX) | [`export/onnx_export.md`](export/onnx_export.md) | ONNX 导出原理：torchvision/ultralytics 规范、预处理对齐、验证 |
| 6 | Export (TensorRT) | [`export/tensorrt_conversion.md`](export/tensorrt_conversion.md) | TensorRT 转换原理：FP16/INT8 量化、校准器策略、决策树 |
| 7 | Export (Triton) | [`export/triton_deployment.md`](export/triton_deployment.md) | Triton 部署配置：模型仓库结构、config.pbtxt 生成、后端对比 |
| 8 | C++ | [`modules/spec_cpp.md`](modules/spec_cpp.md) | C++ 推理模块：独立可提取后端、精度对齐 |

## Reading Order

| 读者 | 推荐顺序 |
|------|----------|
| **开发者 / AI Agent** | [`SSD_AGENT.md`](SSD_AGENT.md) → 按影响范围选择模块 spec |
| **架构理解** | `modules/spec_architecture.md` → `modules/spec_python.md` |
| **Python 开发** | `modules/spec_python.md` → `modules/spec_architecture.md` |
| **模型导出（架构概览）** | `modules/spec_export.md` |
| **模型导出（原理理解）** | `modules/spec_export.md` → `export/onnx_export.md` → `export/tensorrt_conversion.md` → `export/triton_deployment.md` |
| **C++ 部署** | `modules/spec_cpp.md` |

## Task Coverage

| 任务 | Python | C++ | Export |
|------|--------|-----|--------|
| Classification | ✅ | ✅ | ✅ |
| Detection | ✅ | ✅ | ✅ |
| Instance Segmentation | ✅ | ✅ | ✅ |
| Semantic Segmentation | ✅ | ❌ | ✅ |
| Multi-modal (CLIP) | ✅ | ❌ | ✅ |
