# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ModelFlow is a computer vision model deployment toolkit focused on **model evaluation, export, and inference**.
Supports object classification, detection, instance segmentation, semantic segmentation, and multi-modal (CLIP)
across multiple inference backends:

- **ONNX Runtime** (CPU/GPU via CUDA provider)
- **TensorRT** (FP16 and INT8 quantization support)
- **Triton Inference Server** (with ONNX Runtime and TensorRT backends)

Architecture: `InferencePipeline = Preprocessor + Backend + Postprocessor` — a unified, task-driven Pipeline pattern.

## Specifications

The `specs/` directory contains the **canonical specifications** — the single source of truth for SDD Agent development. It is organized into two layers:

```
specs/
├── SDD_METHODOLOGY.md             # Universal SDD methodology (project-agnostic)
├── SDD_GUIDE.md                   # ModelFlow-specific development guide (entry point)
│
├── modules/                       # HOW — internal module architecture & interface contracts
│   ├── spec_architecture.md        # Architecture: all modules + Pipeline pattern
│   ├── spec_python.md              # Python package: ABCs, backends, processors, pipelines
│   ├── spec_eval.md                # eval module: evaluator ABCs, constructor injection, DataFlow-CV bridge
│   ├── spec_export.md              # Export module: pipeline stages, depth levels, validation
│
├── export/                         # WHAT — export format principles & conversion knowledge
│   ├── index.md                    # Export knowledge layer overview
│   ├── onnx_export.md              # ONNX export principles
│   ├── tensorrt_conversion.md      # TensorRT conversion & quantization
│   └── triton_deployment.md        # Triton deployment configuration
│
└── evaluate/                       # WHAT — evaluation bridge contract
    ├── index.md                    # Evaluate layer overview
    └── spec_evaluate_bridge.md     # ModelFlow ↔ DataFlow-CV bridge contract
```

### Specs vs CLAUDE.md

- **Specs** define "what is correct" — the behavioral contract. Stable; change only when requirements change.
- **CLAUDE.md** describes "how the code works" — architecture, patterns, known gotchas. Evolves with the codebase.
- For SDD Agent development, specs are the **compliance benchmark**; CLAUDE.md is the **development context**.
- The SDD methodology is documented in [`specs/SDD_METHODOLOGY.md`](specs/SDD_METHODOLOGY.md) (universal) and [`specs/SDD_GUIDE.md`](specs/SDD_GUIDE.md) (ModelFlow-specific). Start at [`specs/SDD_GUIDE.md`](specs/SDD_GUIDE.md).

### Pre-Change Classification (MANDATORY)

Before creating, modifying, or deleting any file, classify the change against [`specs/SDD_METHODOLOGY.md`](specs/SDD_METHODOLOGY.md) §2.2:

| Priority | Document | Trigger | Action |
|----------|----------|---------|--------|
| **P0** | Specs | Behavior change (interface, contract, data flow) | **Specs first** — update specs, then implement |
| **P1** | CLAUDE.md | New architecture detail, gotcha, hard constraint, or key implementation detail | Sync alongside code |
| **P1** | README | API change, new feature entry point, install step change | Sync user docs |
| **P2** | Samples | User API change, new task type, calling convention change | Update examples |

- Do NOT add usage commands (e.g., `python3 samples/xxx.py`) to CLAUDE.md — those belong in module READMEs.
- Do NOT update CLAUDE.md for P2 sample-script refactors unless they introduce a new gotcha or architecture constraint.

## Common Commands

### Running Tests

```bash
# Run all modelflow tests
pytest modelflow/tests/ -v

# Run all eval tests
pytest eval/tests/ -v

# Run all data tests
pytest data/tests/ -v

# Run a single test file
pytest modelflow/tests/test_processors.py -v

# Run a single test
pytest modelflow/tests/test_processors.py::test_detect_postprocess_nms -v

# Run export tests
pytest export/tests/ -v

# Run full suite (all modules)
pytest modelflow/tests/ eval/tests/ data/tests/ export/tests/ vlms/clip/tests/ -v
```

### Docker Environments

```bash
# TensorRT development (ultralytics/yolov5:v7.0 image)
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/zjykzj:/workdir --workdir=/workdir --name ultra ultralytics/yolov5:v7.0 bash

# Triton server
docker run --gpus=all -it -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:24.06-py3 \
  tritonserver --model-repository=./models/triton/
```

### Utilities

```bash
# Test TensorRT engine
trtexec --loadEngine=models/tensorrt/yolov8s_fp16.engine --iterations=100

# Download COCO dataset
bash assets/get_coco.sh

# Download COCO128 dataset
bash assets/get_coco128.sh
```

## Architecture

### Module Independence (five modules, no cross-dependencies)

| Module | Purpose | Tests |
|--------|---------|-------|
| `utils/` | Shared utilities — Logger, Profile, xywh2xyxy | — |
| `modelflow/` | Python inference pipeline (lazy-import, direct construction) | `tests/` (5 test files) |
| `eval/` | Evaluation orchestration — constructor injection, DataFlow-CV bridge | `tests/` |
| `export/` | PyTorch → ONNX → TensorRT → Triton pipeline | `export/tests/` |
| `data/` | Dataset loading, YAML configs, BaseDataset ABC | — |

### Architecture Constraint Diagram

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
1. **`export/` → `modelflow/`**: Zero dependency. `export/` uses its own self-contained preprocessing in `export/_utils.py`.
2. **Backend → Preprocessor/Postprocessor**: Zero reference. Backends never import or call processors.
3. **Processor → Backend**: Zero direct call. Pipeline is the sole orchestrator.
4. **Pipeline → Backend**: Direct construction via `_build_backend(name, ...)` with lazy imports (`_ensure_backend`). Each backend only needs its own runtime installed.

### modelflow Package Structure

```
modelflow/
├── __init__.py       # Package version + re-exports (ModelInfo, BaseBackend, etc.)
├── interfaces.py     # ABCs: InferencePipeline, BaseBackend, BasePreprocessor, BasePostprocessor
├── types.py          # ModelInfo dataclass
├── config.py         # ModelConfig + COCO (80 classes), ImageNet (1000 classes)
├── backends/         # Inference backends (lazy-import, direct construction)
│   ├── onnx.py           # OnnxBackend (auto CUDA provider selection)
│   ├── tensorrt.py       # TensorrtBackend (CUDA buffer management)
│   └── triton.py         # TritonBackend (gRPC/HTTP)
├── processors/       # Pre/post-processors by task
│   ├── classify/         # Resize+crop+normalize → softmax+top-k
│   ├── detect/           # LetterBox → NMS+box decode (YOLOv5/v8/v11)
│   ├── segment/          # LetterBox → NMS+proto mask decode
│   ├── semantic_seg/     # Resize+normalize → argmax+colormap
├── pipelines/        # Factory functions (_ensure_backend + _build_backend)
│   ├── classify.py
│   ├── detect.py
│   ├── segment.py
│   └── semantic_seg.py
```

### export Package Structure

```
export/
├── _base.py                       # BaseExporter ABC
├── _validation.py                 # ONNX format check + PT vs ONNX comparison
├── _utils.py                      # Self-contained preprocessing (NumPy, zero external deps)
├── onnx/                          # L1: PyTorch → ONNX
│   ├── convert.py                 #   TorchvisionExporter (30+ architectures)
│   ├── ultralytics.py             #   UltralyticsExporter (YOLOv8/v11, detect/seg/cls/pose)
│   └── optimize.py                #   onnx-simplifier wrapper
├── tensorrt/                      # L2: ONNX → TensorRT (FP16 / INT8)
│   ├── build_fp16.py              #   FP16 engine (trtexec + Python API dual path)
│   ├── build_int8.py              #   INT8 engine (PyTorch calibrator)
│   ├── build_int8_pycuda.py       #   INT8 engine (PyCUDA calibrator, Jetson/embedded)
│   ├── calibrator.py              #   BaseCalibrator + TorchCalibrator + PyCudaCalibrator
│   └── scripts/                   #   Calibration data generation
│       ├── generate_calib_cache_for_coco.py
│       ├── generate_calib_cache_for_imagenet.py
│       └── random_copy_images.py
├── triton/                        # L3: Triton deployment config
│   ├── config_generator.py        #   config.pbtxt auto-generation (task-aware)
│   └── model_repo.py              #   Model repository builder + model deployment
└── tests/
    ├── test_export.py
    └── test_engine.py
```

See [`export/README.md`](export/README.md) for the full L1/L2/L3 usage guide.

### Pipeline Factories (Lazy Import + Direct Construction)

All backends are lazy-imported — each only needs its own runtime installed.
Pipeline factories use direct construction via `_build_backend()`:

```python
# modelflow/pipelines/detect.py

def _ensure_backend(name: str) -> None:
    """Lazy-import the requested backend module."""
    if name == "onnxruntime":
        from modelflow.backends.onnx import OnnxBackend  # noqa: F401
    elif name == "tensorrt":
        from modelflow.backends.tensorrt import TensorrtBackend  # noqa: F401
    elif name == "triton":
        from modelflow.backends.triton import TritonBackend  # noqa: F401

def _build_backend(name, model_path, class_list, task_type, half, device):
    """Direct construction — no Registry indirection."""
    if name == "onnxruntime":
        from modelflow.backends.onnx import OnnxBackend
        return OnnxBackend(model_path, class_list, task_type=task_type,
                           half=half, device=device)
    elif name == "tensorrt":
        from modelflow.backends.tensorrt import TensorrtBackend
        return TensorrtBackend(model_path, class_list, task_type=task_type,
                               half=half, device=device)
    elif name == "triton":
        from modelflow.backends.triton import TritonBackend
        return TritonBackend(model_path, class_list, task_type=task_type,
                             half=half, device=device)
    raise ValueError(f"Unsupported backend: {name}")
```

To add a new backend: create the backend file, add entries in `_ensure_backend` and `_build_backend`.
No Registry, no decorators — each backend is independently extractable.

### Key Contracts

- **Backend contract**: takes `np.ndarray` (NCHW float32 preprocessed), returns `List[np.ndarray]` (raw model outputs). No image processing inside backends.
- **Pipeline contract**: `pipeline(image)` runs preprocess → infer → postprocess end-to-end. `pipeline.infer(tensor)` skips pre/post for direct backend access.

### Critical Implementation Details

#### Backend Input/Output Contract

```
Input:  np.ndarray, shape=(N, C, H, W), dtype=float32, range=[-1, 1] or [0, 1]
        (already preprocessed — no image-level transformations in backend)
Output: List[np.ndarray] — raw model outputs, one per output tensor

OnnxBackend:      auto-selects CUDAExecutionProvider if CUDA available
TensorrtBackend:  manages CUDA page-locked host + device buffers; __del__ frees them
TritonBackend:    supports both gRPC (default localhost:8001) and HTTP (8000)
```

#### YOLO Version Differences in Postprocessing

YOLOv5 and YOLOv8/v11 output different tensor shapes, handled by `model_version` param:

| Version | Output Shape | Decode Logic |
|---------|-------------|--------------|
| v5 | `(1, 25200, 85)` | Each row is `[cx, cy, w, h, conf, cls1..cls80]`; decode directly |
| v8/v11 | `(1, 84, 8400)` | Transpose to `(1, 8400, 84)`; `[cx, cy, w, h, conf, cls1..cls80]` |

Both paths converge on the same NMS + coordinate scaling pipeline.

#### Segment Postprocessor Mask Decode

1. Detection output (84 channels: 4 bbox + 80 cls + 32 mask coeffs)
2. Proto mask output: `(1, 32, 160, 160)` — 32 prototype masks
3. `process_mask`: `sigmoid(tensordot(mask_coeffs, proto_masks))` → resize to bbox → crop

#### Triton Model Naming Convention

From `specs/export/triton_deployment.md`:

```
<Task>_<Dataset>_<ModelArch>_<Backend>
Example: Detect_COCO_YOLOv8s_ONNX, Classify_ImageNet_EfficientNetB0_TRT
```

### Export Depth Levels

| Level | Output | Runtime |
|-------|--------|---------|
| L1 | `.onnx` | ONNX Runtime (CPU/GPU) |
| L2 | `.onnx` + `.engine` | TensorRT GPU (FP16/INT8) |
| L3 | `.onnx` + `.engine` + Triton config | Triton Inference Server |

## Development Notes

### Version

Package version: defined in `modelflow/__init__.py` (`__version__`). The CHANGELOG records project history from v0.3.0.

### Environment Setup

No `requirements.txt` or `pyproject.toml` — uses system-wide Python packages. Install dependencies per task:

```bash
# Core
pip install torch torchvision onnx onnxruntime numpy

# Per-backend (install as needed)
pip install tensorrt           # TensorRT
pip install tritonclient[grpc] # Triton client
pip install pycuda             # INT8 PyCUDA calibrator

# Evaluation
pip install pycocotools        # COCO evaluation utilities

# Visualization (samples/infer.py, processors)
pip install opencv-python      # Image I/O in processors
```

### Test Suite (all modules)

All tests use pytest without external fixtures. No model files are required — backends test import/init only (not actual inference).

#### Test Structure

```
modelflow/tests/                 # modelflow inference tests
├── test_core.py                 # ModelInfo, ModelConfig, Interfaces, Pipeline
├── test_backends.py             # OnnxBackend, TensorrtBackend, TritonBackend init
├── test_pipelines.py            # Factory functions reject invalid backends
├── test_processors.py           # 4 processor families + detect ops
└── test_cfgs.py                 # COCO 80 classes, ImageNet 1000 classes

eval/tests/                      # eval module tests
├── test_eval.py                 # BaseEvaluator, BaseMetrics interfaces
└── test_metrics.py              # ClassificationMetrics (confusion matrix)

data/tests/                      # data module tests
└── test_datasets.py             # COCODetection, ClassifyDir, COCOSegment

export/tests/                    # export module tests
├── test_export.py               # Preprocessing, exporters, validation
└── test_engine.py               # TRT calibrators, FP16 build, Triton config

vlms/clip/tests/                 # CLIP processor tests
└── test_processors.py           # CLIPImagePreprocessor, CLIPPostprocessor
```

Test coverage includes:
- **Types**: ModelInfo dataclass
- **Config**: defaults, custom values
- **Interfaces**: ABC subclassing, Pipeline composition, warmup cascade
- **Processors**: shape/type/value constraints across all 4 task families (classify/detect/segment/semantic_seg); NMS correctness; empty-detection edge cases; colormap. CLIP processors in `vlms/clip/`.
- **Metrics**: confusion matrix, perfect/imperfect accuracy, reset
- **Datasets**: empty directories, getitem returns expected structure
- **Eval**: BaseEvaluator/BaseMetrics ABCs, evaluator constructor defaults

### Known Gotchas

1. **Backend must not do image processing**: Pre-processing (resize, normalize, letterbox) must happen in Preprocessor, not in Backend. Backend's sole job is `np.ndarray` in → `List[np.ndarray]` out.
2. **YOLO v5 vs v8/v11 output shape**: The two versions have different output tensor shapes (`(1, 25200, 85)` vs `(1, 84, 8400)`). The `model_version` parameter controls which decode path is used. Mixing them up silently produces garbage results.
3. **Empty detection handling**: Postprocessor must handle the case where no detections pass the confidence threshold. Return empty arrays with correct dtypes (shape `(0, 4)` for boxes, `(0,)` for scores/class_ids), not None.
4. **DataFlow-CV optional dependency**: Detection and segmentation evaluators gracefully degrade when DataFlow-CV is not installed. Always guard `import dataflow` with try/except — never fail hard.
5. **TensorRT device buffer cleanup**: `TensorrtBackend.__del__` frees CUDA device buffers. If the backend object is not properly garbage-collected (e.g., circular references), GPU memory leaks. Use explicit `del backend` or `with` context patterns.
6. **OnnxRuntime provider fallback**: `OnnxBackend` auto-selects CUDAExecutionProvider when CUDA is available. On systems without CUDA, it silently falls back to CPU. Set `providers` explicitly to force a specific provider.
7. **Triton model name confusion**: In `TritonBackend`, `model_path` doubles as the Triton model name (not an actual file path). This differs from OnnxBackend and TensorrtBackend where model_path is a file path.
8. **CLIP mean/std values differ**: CLIP preprocessor (in `vlms/clip/`) uses different mean/std from standard ImageNet normalization. Using ImageNet normalize on CLIP inputs produces incorrect results.
9. **Export module zero-dependency**: `export/` contains its own copy of letterbox and preprocessing utilities in `export/_utils.py`. Never import from `modelflow/` inside `export/`.
10. **NMS NumPy implementation**: The NMS in `modelflow/processors/detect/ops.py` is a pure NumPy implementation.
11. **Triton gRPC vs HTTP**: `TritonBackend` defaults to gRPC (port 8001). HTTP uses port 8000. The `protocol` parameter controls this. Mixing them up silently fails with connection refused.
12. **Segment proto mask dimensions**: The prototype mask output is always `(1, 32, 160, 160)` regardless of input size. The mask decode logic (`process_mask`) must account for this fixed spatial dimension.

### CHANGELOG Policy

- **CHANGELOG.md is a historical release log** — only append new entries when cutting a version.
- **Never edit past entries** — they describe what existed at that version, not current state.
- Refactoring/moving code does NOT retroactively invalidate old changelog entries.

### Git Commits

When creating git commits, use the following format:

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body if needed>

Co-Authored-By: DeepSeek-V4.0 <noreply@deepseek.com>
EOF
)"
```

The Co-Authored-By line is **mandatory** for all commits. The only exception is when the AI model actually powering this session is NOT DeepSeek-V4.0 — in that case, the engineer must confirm the correct model and update the Co-Authored-By line accordingly.

Follow conventional commit style:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
- `style`: Changes that do not affect the meaning of the code
- `perf`: Code change that improves performance
- `chore`: Other changes that don't modify src or test files

The AI model used in this project is **DeepSeek-V4.0**, not Claude Opus.

### Git Ignore

Large model files (`.onnx`, `.engine`, `.plan`, `.trt`) are gitignored — do not commit them.
Calibration data directories (`export/cal_coco_src/`, `export/cal_imagenet_src/`) are gitignored.
Output directories (`output/`, `outputs/`, `models/`) are gitignored.

### Important Considerations

1. **Model paths**: Store models in `models/{runtime,tensorrt,triton}/` matching the backend.
2. **Dynamic shapes**: TensorRT engines require fixed input shapes during conversion; use `--input_shape`.
3. **Calibration data**: INT8 quantization requires calibration datasets (COCO or ImageNet subsets).

## References

- `README.md` — project overview, install guide, and Python API examples
- `specs/SDD_GUIDE.md` — **ModelFlow-specific SDD development guide (start here for development)**
- `specs/modules/` — detailed module design specs
- `specs/export/` — ONNX/TensorRT/Triton knowledge layer
- `export/README.md` — export module documentation
- `modelflow/README.md` — modelflow package documentation
- `samples/README.md` — samples usage guide
- `vlms/clip/samples/README.md` and `vlms/openclip/samples/README.md` — CLIP evaluation docs