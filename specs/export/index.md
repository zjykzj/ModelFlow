# Export 模块知识库

> **Status:** Draft
> **Version:** 0.1
> **前置阅读:** `specs/modules/spec_export.md`（架构总览）

## 定位

本目录是 `specs/modules/spec_export.md` 的**知识层扩展**。架构文档定义 **WHAT**（模块要建什么），本系列文档解释 **WHY & HOW**（为什么这样设计、背后的原理、如何决策）。

## 导出路径总览

```
PyTorch 模型
    │
    ├── torchvision（分类）
    │   └── efficientnet_b0, resnet18, mobilenet_v3, ...
    │
    └── Ultralytics（检测/分割/分类/姿态）
        └── yolov8, yolo11, ...
            │
            ▼
        ┌──────────────────────┐
        │  Stage 1: PT → ONNX  │
        └──────────────────────┘
            │
            ├── L1 停止 ─────────────────▶ ONNX Runtime 推理
            │
            ├── L2 继续 ──▶ TensorRT
            │   ├── FP16 ─────────────────▶ .engine → TRT 推理
            │   └── INT8 ──▶ 校准缓存 ──▶ .engine → TRT 推理
            │
            └── L3 继续 ──▶ Triton 配置生成
                └── config.pbtxt + 仓库结构 ──▶ Triton Server 部署
```

## 文档索引

| 文档 | 解决的问题 | 前置知识 |
|------|-----------|---------|
| [`onnx_export.md`](onnx_export.md) | PT→ONNX 的导出机制、两种模型路径差异、预处理对齐规范 | PyTorch, torchvision, Ultralytics |
| [`tensorrt_conversion.md`](tensorrt_conversion.md) | TensorRT 工作原理解释、FP16/INT8 量化决策、校准器设计选型 | ONNX, TensorRT 基础概念 |
| [`triton_deployment.md`](triton_deployment.md) | Triton 模型仓库结构、config.pbtxt 配置规则、后端选型 | Docker, TensorRT/ONNX |

## 导出深度等级

| 等级 | 产出 | 推理方式 | 适用场景 |
|------|------|---------|---------|
| **L1** | `.onnx` | ONNX Runtime (CPU/GPU) | 快速部署、跨平台兼容、无 GPU 或 GPU 较弱 |
| **L2** | `.onnx` + `.engine` | TensorRT (GPU) | 高性能 GPU 推理、FP16/INT8 优化、生产级延迟要求 |
| **L3** | `.onnx` + `.engine` + Triton 配置 | Triton Server | 服务化部署、多模型管理、动态 batch、A/B 测试 |

**注意**：L1 是 L2/L3 的前置条件，但 L2 不是 L3 的前置条件——Triton 可以直接加载 ONNX 模型，也可以加载 TensorRT 引擎。

## 模型支持矩阵

| 模型来源 | 任务类型 | L1 ONNX | L2 FP16 | L2 INT8 | L3 Triton |
|---------|---------|:-------:|:-------:|:-------:|:---------:|
| torchvision | Classification | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Detection | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Segmentation | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Classification | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Pose | ✅ | ✅ | ✅ | ✅ |

## 量化策略速览

| 量化方式 | 精度损失 | 加速比 | 硬件要求 | 准备工作 |
|---------|---------|-------|---------|---------|
| FP16 | ~0% | 1.5–2× | 支持 FP16 的 GPU (Turing+) | 无需额外数据 |
| INT8 | ~0.5–1% | 2–3× | 支持 INT8 的 GPU (Turing+) | 需 50–100 张校准图片 |

详细对比见 [`tensorrt_conversion.md`](tensorrt_conversion.md#4-量化策略选择)。
