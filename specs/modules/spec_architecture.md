# ModelFlow Architecture

> **Version:** 0.7
> **Status:** Implemented
> **Dependencies:** None (foundation document)

## 1. Module Inventory

ModelFlow consists of independent top-level modules with minimal, unidirectional dependencies:

```
ModelFlow/
├── utils/           # Logger, ops, helpers (Profile), model_info, latency profiling
├── data/            # Dataset loading & YAML configs (numpy + opencv-python + pyyaml)
├── modelflow/       # Pure Python inference engine (pre/post processing + multi-backend inference)
├── eval/            # Evaluation orchestration — pipeline+dataset via constructor injection
├── export/          # Model export (pt → onnx → tensorrt/triton)
└── vlms/            # CLIP / OpenCLIP processors and samples (independent of modelflow)
```

| Module | Responsibility | Key Dependencies |
|--------|----------------|------------------|
| `utils/` | Shared logger, coordinate ops (`xywh2xyxy`), helpers (`Profile`), model metadata (`get_model_info`), latency profiling (`measure_*_latency`, `print_summary`) | numpy, onnx (optional, for model_info) |
| `data/` | Dataset loading, YAML configs (standard datasets per task), BaseDataset ABC | numpy, opencv-python, pyyaml, `utils/` |
| `modelflow/` | Pure inference engine: Pipeline factories, pre/post processors, multi-backend inference (lazy-import, direct construction) | ONNX Runtime, TensorRT, Triton Client (per-backend, all optional), `utils/` |
| `eval/` | Evaluation orchestration — accepts pipeline+dataset via constructor injection (duck-typed, **zero import dependency** on modelflow/ or data/). Delegates mAP to DataFlow-CV. | numpy, tqdm, DataFlow-CV (optional, guarded), `utils/` |
| `export/` | PyTorch → ONNX → TensorRT (FP16/INT8) → Triton config | PyTorch, ONNX, TensorRT — **zero** dependency on modelflow/ |
| `vlms/` | CLIP / OpenCLIP pre/post processors and evaluation samples | Independent of modelflow/ (uses own preprocessing) |

**Key Constraints**:

- Python inference uses only ONNX Runtime / TensorRT / Triton — **does not involve PyTorch inference**
- PyTorch is used **only** within the `export/` module for model export
- Evaluation metrics **depend on [DataFlow-CV](https://github.com/zjykzj/DataFlow-CV)** (with fallback to `{"num_predictions": N}`)
- **`modelflow/` is a pure inference engine** — zero knowledge of evaluation, metrics, or datasets
- **`eval/` has zero import dependency on `modelflow/` or `data/`** — receives pipeline and dataset via constructor injection (duck-typed interfaces)
- **`export/` is fully independent** — contains its own preprocessing in `_utils.py`; never imports from `modelflow/`
- All modules are **independently extractable**; `samples/` is the sole assembly layer

## 2. Core Pattern: Pipeline

```
InferencePipeline = Preprocessor + Backend + Postprocessor
```

This is the core abstraction that runs through the inference modules:

```
image (HWC, BGR) → Preprocessor → tensor (NCHW) → Backend → raw → Postprocessor → result (dict)
```

| Stage | Implementation | Responsibility |
|-------|---------------|----------------|
| **Preprocessor** | NumPy + OpenCV | Image → network input tensor |
| **Backend** | ONNX Runtime / TensorRT / Triton | Tensor → raw inference output |
| **Postprocessor** | NumPy | Raw output → structured result |

## 3. Inter-Module Relationships

```
                       ┌─────────────────────┐
                       │    PyTorch (export)   │
                       └─────────┬───────────┘
                                 │ pt → onnx
                                 ▼
                  ┌──────────────────────────┐
                  │        export/            │
                  │  onnx → tensorrt, triton  │
                  └──────────────────────────┘
                        │           │
                        │ *.onnx    │ *.engine / triton config
                        ▼           ▼
 ┌──────────────────────────────────────────┐
 │              modelflow/                   │
 │  Pipeline(preprocessor + backend + post)  │
 │  Pure inference engine                    │
 └──────────────────────────────────────────┘

 ┌──────────────────────────────────────────┐
 │              eval/                        │
 │  Evaluator(pipeline, dataset) → run()    │
 │  └──▶ DataFlow-CV (guarded import)       │
 └──────────────────────────────────────────┘

 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │   utils/      │  │    data/      │  │    vlms/      │
 │ logger, ops,  │  │ Dataset+YAML  │  │ CLIP/OpenCLIP │
 │ model_info,   │  │               │  │               │
 │ profile       │  │               │  │               │
 └──────────────┘  └──────────────┘  └──────────────┘
```

## 4. Data Flow

### 4.1 Inference Data Flow

```
User input (image path / cv2 image)
        │
        ▼
Preprocessor.__call__(image)
    ├── NumPy:  letterbox/resize → HWC→CHW → normalize
    └── TorchVision: torchvision.transforms
        │
        ▼  Input Tensor (NCHW, float32)
Backend.__call__(tensor)
    ├── ONNX:   session.run()
    ├── TensorRT: execute_async_v2()
    └── Triton:  grpc/http infer()
        │
        ▼  Raw Outputs (List[np.ndarray])
Postprocessor.__call__(raw, **kwargs)
    ├── classify:  softmax → top-k
    ├── detect:    NMS → scale_boxes → (boxes, scores, labels)
    ├── segment:   NMS → process_mask → (boxes, scores, labels, masks)
    └── semantic:  argmax → colormap
        │
        ▼  Structured Result (dict)
```

### 4.2 Evaluation Data Flow

```
samples/ (assembly layer)                    eval/ (orchestration)
────────────────────────────────     ─────────────────────────────
Dataset → Pipeline → predictions ──▶ Evaluator.run()
                                         │
                                         ├── Classify: local ClassificationMetrics
                                         ├── Detect:  delegate to DataFlow-CV DetectionEvaluator
                                         └── Segment: delegate to DataFlow-CV SegmentationEvaluator
```

### 4.3 Model Analysis Data Flow (no dataset)

Two standalone scripts for the two phases of model analysis:

```
Phase 1: samples/analyze_model.py  (architecture, ONNX only)
─────────────────────────────────────────────────────────────
Model file → get_model_info() → size, params, FLOPs, I/O shapes
                                → infer_config() → task_type, model_version, num_classes

Phase 2: samples/bench_model.py   (latency, any backend)
─────────────────────────────────────────────────────────────
Model file → build_pipeline() → measure_*_latency() → mean/p95 ms, FPS
                               backend-only mode: dummy tensor, pure inference timing
                               pipeline mode: real image, pre + infer + post stages
```

## 5. Task Coverage Matrix

| Task Type | Python Pipeline | Export | Evaluator (eval/) |
|-----------|:---------------:|:------:|:-----------------:|
| Classification | ✅ | ✅ | Local ClassificationMetrics |
| Detection | ✅ | ✅ | DataFlow-CV DetectionEvaluator |
| Instance Segmentation | ✅ | ✅ | DataFlow-CV SegmentationEvaluator |
| Semantic Segmentation | ✅ | ✅ | Not yet implemented |
| Multi-modal (CLIP) | ✅ (in `vlms/`) | ✅ | Not yet implemented |

## 6. Implementation Roadmap

| Phase | Content | Deliverables |
|-------|---------|-------------|
| 1 | Infrastructure | interfaces, types, config, Pipeline core |
| 2 | Classification inference + evaluation | OnnxBackend, ClassifyProcessor, ClassifyEvaluator |
| 3 | Detection inference + evaluation | TensorrtBackend, TritonBackend, DetectProcessor, DataFlow-CV bridge |
| 4 | Segmentation + semantic segmentation | SegmentProcessor, SemanticSegProcessor |
| 5 | Multi-modal | CLIP Processor (in `vlms/`) |
| 6 | Export module | pt→onnx, onnx→trt, triton config |
| 7 | Model analysis tooling | Model metadata collection, latency profiling, standalone parse script |

## 7. Module Dependency Contract

### 7.1 Cross-Module Import Rules

```
utils/
├── May import: numpy, onnx (optional, for model_info), logging (standard lib)
└── Must NOT import: any ModelFlow module

modelflow/
├── May import: numpy, onnxruntime, tensorrt, tritonclient, torch, cv2, PIL
├── May import: utils.*
├── Must NOT import: dataflow.*, pycocotools (evaluation belongs in eval/ only)
├── Must NOT import: export/*, data/*, eval/*, vlms/*
└── ZERO dependency on: export/, data/, eval/, vlms/

eval/
├── May import: numpy, cv2, tqdm, json, os
├── May import: utils.* (logger, ops)
├── May import (guarded): dataflow.evaluate, dataflow.util.logging
├── Must NOT import: modelflow/*, data/* (duck-typed — constructor injection only)
└── Must NOT import: export/* (architecture constraint)

export/
├── May import: torch, torchvision, onnx, tensorrt, numpy
├── May import: its own _utils.py (self-contained preprocessing)
└── Must NOT import: modelflow/*, data/*, eval/*, utils/*

vlms/
├── May import: torch, numpy, cv2, PIL
├── May import: utils.*
└── Must NOT import: modelflow/*, eval/*, export/*
```

### 7.2 Pipeline Data Flow Contract

```
Preprocessor(image: HWC uint8 BGR) → tensor: NCHW float32
Backend(tensor: NCHW float32) → raw: List[np.ndarray]
Postprocessor(raw: List[np.ndarray]) → result: Dict

Evaluator (in eval/): constructor(pipeline, dataset) → run() → Dict[str, float]
```

### 7.3 External Dependency Contract

| External | Used By | Constraint |
|----------|---------|------------|
| **DataFlow-CV** | `eval/` | Guarded import. Fallback to `{"num_predictions": N}` if unavailable. |
| **PyTorch** | `export/` only | Not used for inference in `modelflow/` (processors use NumPy by default) |
| **ONNX Runtime** | `modelflow/backends/onnx.py` | Primary inference backend |
| **TensorRT** | `modelflow/backends/tensorrt.py`, `export/tensorrt/` | GPU-accelerated inference |
