# Export Module Knowledge Base

> **Status:** Implemented
> **Version:** 0.1
> **Prerequisite reading:** `specs/modules/spec_export.md` (architecture overview)

## Purpose

This directory serves as the **knowledge layer** for `specs/modules/spec_export.md`. The architecture document defines **WHAT** the module builds; this series of documents explains the **WHY & HOW** — the design rationale, underlying principles, and decision-making guidance.

## Export Pipeline Overview

```
PyTorch Model
    │
    ├── torchvision (classification)
    │   └── efficientnet_b0, resnet18, mobilenet_v3, ...
    │
    └── Ultralytics (detection/segmentation/classification/pose)
        └── yolov8, yolo11, ...
            │
            ▼
        ┌──────────────────────┐
        │  Stage 1: PT → ONNX  │
        └──────────────────────┘
            │
            ├── Stop at L1 ───────────────▶ ONNX Runtime inference
            │
            ├── Continue to L2 ──▶ TensorRT
            │   ├── FP16 ─────────────────▶ .engine → TRT inference
            │   └── INT8 ──▶ calibration cache ──▶ .engine → TRT inference
            │
            └── Continue to L3 ──▶ Triton config generation
                └── config.pbtxt + repository structure ──▶ Triton Server deployment
```

## Document Index

| Document | Problem it solves | Prerequisite knowledge |
|----------|-------------------|------------------------|
| [`onnx_export.md`](onnx_export.md) | PT→ONNX export mechanics, differences between the two model paths, preprocessing alignment spec | PyTorch, torchvision, Ultralytics |
| [`tensorrt_conversion.md`](tensorrt_conversion.md) | How TensorRT works under the hood, FP16/INT8 quantization trade-offs, calibrator design choices | ONNX, basic TensorRT concepts |
| [`triton_deployment.md`](triton_deployment.md) | Triton model repository structure, config.pbtxt configuration rules, backend selection | Docker, TensorRT/ONNX |

## Export Depth Levels

| Level | Output | Inference runtime | Use case |
|-------|--------|-------------------|----------|
| **L1** | `.onnx` | ONNX Runtime (CPU/GPU) | Rapid deployment, cross-platform compatibility, no GPU or limited GPU |
| **L2** | `.onnx` + `.engine` | TensorRT (GPU) | High-performance GPU inference, FP16/INT8 optimization, production latency requirements |
| **L3** | `.onnx` + `.engine` + Triton config | Triton Server | Serving deployment, multi-model management, dynamic batching, A/B testing |

**Note**: L1 is a prerequisite for L2/L3, but L2 is not a prerequisite for L3 — Triton can load ONNX models directly as well as TensorRT engines.

## Model Support Matrix

| Model source | Task type | L1 ONNX | L2 FP16 | L2 INT8 | L3 Triton |
|-------------|-----------|:-------:|:-------:|:-------:|:---------:|
| torchvision | Classification | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Detection | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Segmentation | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Classification | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Pose | ✅ | ✅ | ✅ | ✅ |

## Quantization Strategy at a Glance

| Method | Accuracy loss | Speedup | Hardware requirement | Preparation |
|--------|--------------|---------|---------------------|-------------|
| FP16 | ~0% | 1.5–2x | FP16-capable GPU (Turing+) | No additional data required |
| INT8 | ~0.5–1% | 2–3x | INT8-capable GPU (Turing+) | 50–100 calibration images needed |

For a detailed comparison, see [`tensorrt_conversion.md`](tensorrt_conversion.md#5-quantization-strategy-selection).
