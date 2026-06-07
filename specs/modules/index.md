# Modules Layer — Specification Index

> **Status:** Draft — these documents define the implementation architecture of ModelFlow.

## What This Layer Covers

The Modules layer defines **HOW** ModelFlow is built. It describes the implementation contracts for:

- The three top-level modules: `modelflow/` (Python), `export/` (model export), `cpp/` (C++ inference)
- The Pipeline pattern that unifies all inference across languages and backends
- Each module's internal structure, interfaces, extension points, and design rationale

## Architecture Diagram

```
                        ┌─────────────────────┐
                        │    PyTorch (导出)     │
                        └─────────┬───────────┘
                                  │ pt → onnx
                                  ▼
                   ┌──────────────────────────┐
                   │        export /           │
                   │  onnx → tensorrt / triton │
                   │                          │
                   │  specs/export/ (知识层)    │
                   │  ├── onnx_export.md      │
                   │  ├── tensorrt_conversion  │
                   │  └── triton_deployment   │
                   └──────────────────────────┘
                         │           │
                         │ *.onnx    │ *.engine / triton config
                         ▼           ▼
  ┌──────────────────────────────────────────┐
  │              modelflow /                  │
  │  Pipeline(processor + backend + postproc) │
  │  ┌────────┐  ┌────────┐  ┌────────────┐  │
  │  │onnx    │  │trt     │  │triton      │  │
  │  └────────┘  └────────┘  └────────────┘  │
  │                                          │
  │  Evaluator ──▶ DataFlow-CV (metrics)     │
  │  Visualizer ─▶ DataFlow-CV (drawing)     │
  └──────────────────────────────────────────┘

  ┌──────────────────────────────────────────┐
  │              cpp /                        │
  │  core/ + backends/onnx/          ── CPU  │
  │  core/ + backends/tensorrt/      ── GPU  │
  │  (各后端可独立提取)                        │
  └──────────────────────────────────────────┘
```

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`spec_architecture.md`](spec_architecture.md) | 三层独立模块设计、模块间关系、Pipeline 模式、数据流、实施路线 |
| 2 | [`spec_python.md`](spec_python.md) | `modelflow/` 包：抽象基类、推理后端、处理器、Pipeline 工厂、评估器、注册机制 |
| 3 | [`spec_export.md`](spec_export.md) | `export/` 模块：pt → onnx → tensorrt/triton 导出管线阶段、I/O 规格、精度验证标准 |
| 4 | [`spec_cpp.md`](spec_cpp.md) | `cpp/` 模块：独立可提取后端、OpenCV 预处理、Pipeline 示例、Python↔C++ 精度对齐 |

> **导出知识层补充：** `spec_export.md` 仅定义架构蓝图。格式原理、转换规范、模型差异和选型依据详见 [`specs/export/`](../export/index.md) 系列文档（[`onnx_export.md`](../export/onnx_export.md)、[`tensorrt_conversion.md`](../export/tensorrt_conversion.md)、[`triton_deployment.md`](../export/triton_deployment.md)）。

## Dependencies Between Modules

```
spec_architecture.md  (模块间关系、Pipeline 模式定义)
    │
    ├──▶ spec_python.md   (实现 Python 侧的 Pipeline 约定)
    ├──▶ spec_export.md   (导出模块架构蓝图)
    │       │
    │       └──▶ specs/export/ 系列 (原理性知识层扩展)
    └──▶ spec_cpp.md      (独立模块，共享 Pipeline 语义，精度对齐需参考 Python 结果)
```

## Reading Order

| 读者 | 推荐顺序 |
|------|----------|
| **全局架构理解** | `spec_architecture.md` → `spec_python.md` |
| **Python 开发** | `spec_python.md` → `spec_architecture.md` |
| **模型导出（架构概览）** | `spec_export.md` |
| **模型导出（原理理解）** | `spec_export.md` → [`specs/export/`](../export/index.md) 系列 |
| **C++ 部署** | `spec_cpp.md` |
