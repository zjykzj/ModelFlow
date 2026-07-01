<div align="center">
  <a href="https://github.com/zjykzj/ModelFlow">
    <img src="./assets/logos/ModelFlow.svg" alt="ModelFlow" width="600">
  </a>
</div>

> 🚀 **CV model deployment toolkit: export → infer → eval, across ONNX Runtime, TensorRT, and Triton.**

<p align="center">
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt="Standard Readme"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt="Conventional Commits"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <br>
  <img src="https://img.shields.io/badge/ONNX%20Runtime-CPU%2FGPU-005ced?logo=onnx" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/TensorRT-FP16%2FINT8-76b900?logo=nvidia" alt="TensorRT">
  <img src="https://img.shields.io/badge/Triton-Server-ff6f00?logo=nvidia" alt="Triton">
</p>

---

## 1. Overview

ModelFlow is a computer vision model deployment toolkit — export PyTorch checkpoints, run inference across backends, and evaluate accuracy — all through a unified Pipeline API.

### Task × Backend Matrix

| Task | ONNX Runtime | TensorRT | Triton |
|------|:---:|:---:|:---:|
| **Classification** | ✓ | ✓ | ✓ |
| **Object Detection** | ✓ | ✓ | ✓ |
| **Instance Segmentation** | ✓ | ✓ | ✓ |
| **Semantic Segmentation** | ✓ | ✓ | ✓ |
| **Multi-modal (CLIP)** | ✓ | — | — |

### Architecture at a Glance

```
InferencePipeline = Preprocessor + Backend + Postprocessor

  image → [Preprocessor] → tensor → [Backend] → raw outputs → [Postprocessor] → result
```

- **Preprocessor**: resize, normalize, letterbox — image → NCHW float32 tensor
- **Backend**: pure inference — `np.ndarray` in, `List[np.ndarray]` out (no image processing)
- **Postprocessor**: softmax/NMS/mask decode — raw outputs → structured result dict

---

## 2. Installation

No `requirements.txt` or `pyproject.toml` — install per use case.

### Core

```bash
pip install torch torchvision onnx onnxruntime numpy
```

### Per Backend (install as needed)

```bash
pip install tensorrt                       # TensorRT GPU
pip install tritonclient[grpc]             # Triton client
pip install pycuda                         # INT8 PyCUDA calibrator (Jetson/embedded)
```

### Evaluation

```bash
pip install pycocotools                    # COCO mAP computation
```

### Visualization

```bash
pip install opencv-python                  # Image I/O, drawing overlays
```

---

## 3. Quick Start

### 3.1 Classification

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

### 3.2 Object Detection

```python
from modelflow.pipelines import create_detect_pipeline
from modelflow.config import COCO_CLASSES

pipeline = create_detect_pipeline(
    model_path="yolov8s.onnx",
    class_list=COCO_CLASSES,
    backend="onnxruntime",
)

result = pipeline(image, conf_thres=0.25, iou_thres=0.45)
# result = {"boxes": ndarray(N,4), "scores": ndarray(N,), "class_ids": ndarray(N,)}
```

### 3.3 Instance Segmentation

```python
from modelflow.pipelines import create_segment_pipeline
from modelflow.config import COCO_CLASSES

pipeline = create_segment_pipeline(
    model_path="yolov8s-seg.onnx",
    class_list=COCO_CLASSES,
)

result = pipeline(image)
# result = {"boxes": ..., "scores": ..., "class_ids": ..., "masks": ...}
```

### 3.4 Semantic Segmentation

```python
from modelflow.pipelines import create_semantic_seg_pipeline

pipeline = create_semantic_seg_pipeline(
    model_path="segformer.onnx",
)

result = pipeline(image)
# result = {"class_map": ndarray(H,W), "colormap": ndarray(H,W,3)}
```

---

## 4. Architecture

### 4.1 Module Independence

Five modules, zero cross-dependencies:

| Module | Purpose | Tests | Documentation |
|--------|---------|-------|---------------|
| `utils/` | Shared utilities — Logger, Profile, xywh2xyxy | — | — |
| `modelflow/` | Python inference pipeline (lazy-import, direct construction) | `tests/` (5 files) | [`modelflow/README.md`](modelflow/README.md) |
| `export/` | PyTorch → ONNX → TensorRT → Triton export pipeline | `export/tests/` | [`export/README.md`](export/README.md) |
| `eval/` | Evaluation orchestration — constructor injection, DataFlow-CV bridge | `tests/` | [`eval/README.md`](eval/README.md) |
| `data/` | Dataset loading, YAML configs, BaseDataset ABC | `tests/` | [`data/README.md`](data/README.md) |

### 4.2 Constraint Diagram

```
                    ┌─────────────────┐
                    │     samples/     │
                    │ (infer / eval)   │
                    └──────┬──────────┘
                           │  calls pipeline factories
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         modelflow/                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
│  │pipelines │──>│processors│   │ backends │                    │
│  │(factory) │   │(pre/post)│   │(infer)   │                    │
│  └──────────┘   └──────────┘   └──────────┘                    │
│       │              │              │                            │
│       │              │   ZERO      │                            │
│       │              │   CROSS-    │                            │
│       │              │   DEPENDENCY│                            │
│       │              │              │                            │
│       └──────┬───────┘              │                            │
│              │                      │                            │
│              ▼                      ▼                            │
│  interfaces.py + types.py + config.py                              │
│  (flattened — no core/ subpackage)                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       export/ (ZERO dependency on modelflow/)    │
│  _base/_validation/_utils → onnx/ → tensorrt/ → triton/         │
└─────────────────────────────────────────────────────────────────┘
```

**Hard constraints:**

1. **`export/` → `modelflow/`**: Zero dependency. `export/` ships its own preprocessing in `export/_utils.py`.
2. **Backend → Preprocessor/Postprocessor**: Zero reference. Backends never import or call processors.
3. **Processor → Backend**: Zero direct call. Pipeline is the sole orchestrator.
4. **Pipeline → Backend**: Direct construction via `_build_backend(name, ...)` with lazy imports — each backend only needs its own runtime installed.

### 4.3 Backend Input/Output Contract

```
Input:  np.ndarray, shape=(N, C, H, W), dtype=float32
        (preprocessed — no image-level ops in backend)
Output: List[np.ndarray] — raw model outputs, one per output tensor

OnnxBackend:      auto-selects CUDAExecutionProvider if CUDA available
TensorrtBackend:  manages CUDA page-locked host + device buffers
TritonBackend:    gRPC (default localhost:8001) and HTTP (8000)
```

---

## 5. Module Map

```
ModelFlow/
├── modelflow/        Python inference pipeline — Pipeline = Preprocessor + Backend + Postprocessor
│   ├── interfaces.py   ABCs: InferencePipeline, BaseBackend, BasePreprocessor, BasePostprocessor
│   ├── backends/       OnnxBackend / TensorrtBackend / TritonBackend (lazy-import)
│   ├── processors/     Pre/post-processors by task: classify / detect / segment / semantic_seg
│   └── pipelines/      Factory functions: create_classify_pipeline, create_detect_pipeline, ...
│
├── export/           Model export pipeline — three depth levels
│   ├── onnx/           L1: PT → ONNX (torchvision + Ultralytics)
│   ├── tensorrt/       L2: ONNX → TensorRT (FP16 / INT8)
│   └── triton/         L3: Triton config + model repository
│
├── eval/             Evaluation orchestration (depends on modelflow/ + data/)
│   ├── evaluators/     ClassifyEvaluator / DetectEvaluator / SegmentEvaluator
│   └── metrics/        ClassificationMetrics (confusion matrix → Accuracy/Precision/Recall/F1)
│
├── data/             Dataset loading — YAML-driven, zero dependency on the rest of ModelFlow
│   ├── configs/        Built-in YAML configs: coco, coco-seg, imagenet
│   └── build.py        Factory: build_dataset, get_class_names, load_config
│
├── vlms/             Vision-Language Models — multi-modal (image + text)
│   ├── clip/           OpenAI CLIP (pre/post processors + evaluation)
│   └── openclip/       OpenCLIP (evaluation samples)
│
├── samples/          Runnable examples — infer, eval, analyze, bench
├── utils/            Shared utilities — Logger, Profile, coordinate transforms
└── specs/            Architecture & interface specifications (canonical design docs)
```

> Each sub-package has its own README with detailed API docs and usage examples.

---

## 6. Export Pipeline

Three independent depth levels — stop at any stage:

| Level | Output | Runtime | Key Feature |
|-------|--------|---------|-------------|
| **L1** | `.onnx` | ONNX Runtime (CPU/GPU) | Cross-platform inference |
| **L2** | `.onnx` + `.engine` | TensorRT GPU | FP16 (1.5–2× speedup) / INT8 (2–3× speedup) |
| **L3** | `.onnx` + `.engine` + Triton config | Triton Inference Server | Production serving (gRPC/HTTP) |

### Quick Export Commands

```bash
# L1 — Export to ONNX
python3 -m export.onnx.ultralytics yolov8s --save models/runtime/yolov8s.onnx
python3 -m export.onnx.convert --model efficientnet_b0 --save models/runtime/efficientnet_b0.onnx

# L2 — Build TensorRT Engine
python3 -m export.tensorrt.build_fp16 --onnx models/runtime/yolov8s.onnx --save models/tensorrt/yolov8s_fp16.engine

# L3 — Generate Triton Config
python3 -m export.triton.config_generator --model-name Detect_COCO_YOLOv8s_ONNX --backend onnxruntime --task detect
```

> See **[`export/README.md`](export/README.md)** for the full L1/L2/L3 reference with calibration guides and Python API.

---

## 7. Evaluation

### Evaluator Types

| Evaluator | Task | Metrics | DataFlow-CV |
|-----------|------|---------|:---:|
| `ClassifyEvaluator` | Classification | Accuracy, Precision, Recall, F1 (confusion matrix) | — |
| `DetectEvaluator` | Object Detection | mAP, AP50, AP75, … | Required |
| `SegmentEvaluator` | Instance Segmentation | segm mAP, AP50, AP75, … | Required |

### Quick Eval Commands

```bash
# Classification (ImageNet)
python3 samples/eval_classify.py --model models/runtime/efficientnet_b0.onnx --data /path/to/imagenet

# Detection (COCO mAP)
python3 samples/eval_detect.py --model models/runtime/yolov8s.onnx --data /path/to/coco
python3 samples/eval_detect.py --model models/tensorrt/yolov8s_fp16.engine --backend tensorrt --data /path/to/coco

# Segmentation (COCO segm mAP)
python3 samples/eval_segment.py --model models/runtime/yolov8s-seg.onnx --data /path/to/coco
```

> Detection/Segment evaluators delegate mAP to DataFlow-CV ([bridge contract](specs/evaluate/spec_evaluate_bridge.md)). Falls back gracefully if DataFlow-CV is not installed — never crashes. See **[`eval/README.md`](eval/README.md)** for the full API.

---

## 8. Samples

The `samples/` directory provides runnable entry points for the full workflow:

### Inference (single image + visualization)

```bash
python3 samples/infer.py --task detect --model models/runtime/yolov8s.onnx --image assets/bus.jpg
python3 samples/infer.py --task classify --model models/runtime/efficientnet_b0.onnx --image assets/cat.jpg
python3 samples/infer.py --task segment --model models/runtime/yolov8s-seg.onnx --image assets/bus.jpg
```

### Model Analysis (three-stage pipeline)

| Stage | Script | What | Time |
|-------|--------|------|------|
| 1 | `analyze_model.py` | Params, FLOPs, I/O shape, task inference | ms |
| 2 | `bench_model.py` | Backend/pipeline latency (ONNX/TRT/Triton) | seconds |
| 3 | `eval_*.py` | Accuracy — mAP / top-k (ONNX/TRT/Triton) | minutes–hours |

```bash
# Stage 1 — Architecture analysis
python3 samples/analyze_model.py --model models/yolov8s.onnx

# Stage 2 — Latency benchmark
python3 samples/bench_model.py --model models/yolov8s.onnx --task detect

# Stage 3 — Accuracy validation
python3 samples/eval_detect.py --model models/yolov8s.onnx --data /path/to/coco
```

> See **[`samples/README.md`](samples/README.md)** for the full command reference across all stages and backends.

---

## 9. References

| Document | Content |
|----------|---------|
| [`modelflow/README.md`](modelflow/README.md) | Python inference package — Pipeline API, backends, processors |
| [`export/README.md`](export/README.md) | Export pipeline — L1/L2/L3 full reference |
| [`eval/README.md`](eval/README.md) | Evaluation module — evaluators, metrics, DataFlow-CV bridge |
| [`data/README.md`](data/README.md) | Data module — YAML configs, datasets, factory API |
| [`samples/README.md`](samples/README.md) | Sample scripts — infer, eval, analyze, bench |
| [`vlms/README.md`](vlms/README.md) | Vision-Language Models — CLIP, OpenCLIP |
| [`specs/SDD_GUIDE.md`](specs/SDD_GUIDE.md) | SDD development guide (entry point for contributors) |
| [`specs/modules/`](specs/modules/) | Module architecture specifications |

---

## 10. License

[Apache License 2.0](LICENSE) © 2026 zjykzj
