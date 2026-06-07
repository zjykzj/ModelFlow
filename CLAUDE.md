# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ModelFlow is a computer vision model deployment toolkit focused on **model evaluation, export, and inference**.
Supports object classification, detection, instance segmentation, semantic segmentation, and multi-modal (CLIP)
across multiple inference backends:

- **ONNX Runtime** (CPU/GPU via CUDA provider)
- **TensorRT** (FP16 and INT8 quantization support)
- **Triton Inference Server** (with ONNX Runtime and TensorRT backends)

Architecture: `InferencePipeline = Preprocessor + Backend + Postprocessor` — a unified, task-driven Pipeline pattern
handling everything from image loading to metric computation.

## Specifications

The `specs/` directory contains the **canonical specifications** — the single source of truth for SDD Agent development. It is organized into two layers:

```
specs/
├── modules/                        # HOW — internal module architecture & interface contracts
│   ├── spec_architecture.md        # Architecture: three modules + Pipeline pattern
│   ├── spec_python.md              # Python package: ABCs, backends, processors, evaluators
│   ├── spec_export.md              # Export module: pipeline stages, depth levels, validation
│   └── spec_cpp.md                 # C++ inference module (planned)
│
└── export/                         # WHAT — export format principles & conversion knowledge
    ├── index.md                    # Export knowledge layer overview
    ├── onnx_export.md              # ONNX export principles
    ├── tensorrt_conversion.md      # TensorRT conversion & quantization
    └── triton_deployment.md        # Triton deployment configuration
```

### Specs vs CLAUDE.md

- **Specs** define "what is correct" — the behavioral contract. Stable; change only when requirements change.
- **CLAUDE.md** describes "how the code works" — architecture, patterns, known gotchas. Evolves with the codebase.
- For SDD Agent development, specs are the **compliance benchmark**; CLAUDE.md is the **development context**.
- The full SDD Agent development workflow is documented in [`specs/SDD_AGENT.md`](specs/SDD_AGENT.md).

## Common Commands

### Running Tests

```bash
# Run all modelflow tests (96 tests)
pytest tests/ -v

# Run a single test file
pytest tests/test_processors.py -v

# Run a single test
pytest tests/test_processors.py::test_detect_postprocess_nms -v

# Run export tests
pytest export/tests/ -v

# Run full suite including export
pytest tests/ export/tests/ -v
```

### Model Inference (unified)

The recommended entry point is the unified `samples/infer.py` (replaces 15 legacy scripts):

```bash
# Detection (ONNX Runtime)
python3 samples/infer.py --task detect --model models/runtime/yolov8s.onnx --image assets/bus.jpg

# Detection (TensorRT)
python3 samples/infer.py --task detect --model models/tensorrt/yolov8s_fp16.engine --image assets/bus.jpg --backend tensorrt

# Detection (Triton)
python3 samples/infer.py --task detect --model Detect_COCO_YOLOv8s_ONNX --image assets/bus.jpg --backend triton

# YOLOv5 detection
python3 samples/infer.py --task detect --model models/runtime/yolov5s.onnx --image assets/bus.jpg --model-version v5

# Classification
python3 samples/infer.py --task classify --model models/runtime/efficientnet_b0.onnx --image assets/bus.jpg --classes imagenet --input-size 224

# Instance segmentation
python3 samples/infer.py --task segment --model models/runtime/yolov8s-seg.onnx --image assets/bus.jpg
```

### Model Evaluation (unified)

The consolidated `samples/eval_bench.py` (replaces 14 legacy benchmark scripts):

```bash
# Detection evaluation
python3 samples/eval_bench.py --task detect --model models/runtime/yolov8s.onnx --data /path/to/coco/val2017

# Classification evaluation
python3 samples/eval_bench.py --task classify --model models/runtime/efficientnet_b0.onnx --data /path/to/val --classes imagenet

# TensorRT detection evaluation
python3 samples/eval_bench.py --task detect --model models/tensorrt/yolov8s_fp16.engine --backend tensorrt --data /path/to/coco

# With ground truth annotations for mAP
python3 samples/eval_bench.py --task detect --model models/runtime/yolov8s.onnx --data /path/to/coco --anno-json /path/to/annotations.json
```

### Model Export and Conversion

```bash
# PT → ONNX (torchvision classification)
python3 -m export.onnx.convert --model efficientnet_b0 --save models/runtime/efficientnet_b0.onnx

# PT → ONNX (Ultralytics YOLOv8)
python3 -m export.onnx.ultralytics yolov8s --save models/runtime/yolov8s.onnx

# ONNX → TensorRT FP16
python3 -m export.tensorrt.build_fp16 --onnx models/runtime/yolov8s.onnx --save models/tensorrt/yolov8s_fp16.engine

# ONNX → TensorRT INT8
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir export/cal_coco_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640

# Generate Triton config
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt --task detect --save ./models/triton/
```

### Docker Environments

```bash
# TensorRT development (ultralytics/yolov5:v7.0 image)
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/zjykzj:/workdir --workdir=/workdir --name ultra ultralytics/yolov5:v7.0 bash

# Triton server
docker run --gpus=all -it -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=./models/triton/
```

### Utilities

```bash
# Generate INT8 calibration cache from ImageNet
python3 export/scripts/generate_calib_cache_for_imagenet.py --input_dir cal_src --output_dir cal_dst

# Test TensorRT engine
trtexec --loadEngine=models/tensorrt/yolov8s_fp16.engine --iterations=100

# Download COCO dataset
bash assets/get_coco.sh

# Download COCO128 dataset
bash assets/get_coco128.sh
```

## Architecture

### Module Independence (three sibling modules, no cross-dependencies)

| Module | Purpose | Tests |
|--------|---------|-------|
| `modelflow/` | Python inference, evaluation, visualization | `tests/` (96 tests) |
| `export/` | PyTorch → ONNX → TensorRT → Triton pipeline | `export/tests/` |
| `cpp/` | C++ inference (OpenCV + ONNX Runtime / TensorRT) | — |

### Architecture Constraint Diagram

```
                    ┌─────────────────┐
                    │     samples/     │
                    │ (infer / eval)   │
                    └──────┬──────────┘
                           │  calls pipeline factories & evaluators
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         modelflow/                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│  │pipelines │──>│processors│   │ backends │   │ evaluators   │  │
│  │(factory) │   │(pre/post)│   │(infer)   │   │ (metrics)    │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘  │
│       │              │              │               │            │
│       │              │   ZERO      │               │            │
│       │              │   CROSS-    │               │            │
│       │              │   DEPENDENCY│               │            │
│       │              │              │               │            │
│       └──────┬───────┘              │               │            │
│              │                      │               │            │
│              ▼                      ▼               ▼            │
│         ┌────────────────────────────────────────────────────┐   │
│         │                    core/                            │   │
│         │  interfaces.py + registry.py + types.py + config.py │   │
│         └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       export/ (ZERO dependency on modelflow/)    │
│  core/ → onnx/ → tensorrt/ → triton/                            │
└─────────────────────────────────────────────────────────────────┘
```

**Hard constraints:**
1. **`export/` → `modelflow/`**: Zero dependency. `export/` uses its own self-contained preprocessing in `export/core/utils.py`.
2. **Backend → Preprocessor/Postprocessor**: Zero reference. Backends never import or call processors.
3. **Processor → Backend**: Zero direct call. Pipeline is the sole orchestrator.
4. **Evaluator → Pipeline**: Only through pipeline's public interface (`pipeline(image)` / `pipeline.infer(tensor)`).
5. **pipelines/ → core/**: Factories import ABCs and types from core, then compose processor + backend + postprocessor.

### modelflow Package Structure

```
modelflow/
├── core/           # ABCs, Registry, Types, ModelConfig
│   ├── interfaces.py   # 7 base classes + InferencePipeline composition
│   ├── registry.py     # Registry + 5 global singletons
│   ├── types.py        # TaskType, BackendType, ProcessorType enums
│   └── config.py       # ModelConfig dataclass
├── backends/       # Inference backends
│   ├── onnx.py         # OnnxBackend (auto CUDA provider selection)
│   ├── tensorrt.py     # TensorrtBackend (CUDA buffer management)
│   └── triton.py       # TritonBackend (gRPC/HTTP)
├── processors/     # Pre/post-processors by task
│   ├── classify/       # Resize+crop+normalize → softmax+top-k
│   ├── detect/         # LetterBox → NMS+box decode (YOLOv5/v8/v11)
│   ├── segment/        # LetterBox → NMS+proto mask decode
│   ├── semantic_seg/   # Resize+normalize → argmax+colormap
│   └── multimodal/     # CLIP preprocessing → similarity ranking
├── pipelines/      # Factory functions (classify/detect/segment/semantic_seg)
├── datasets/       # COCODetection, ClassifyDir, COCOSegment
├── evaluators/     # ClassifyEvaluator, DetectEvaluator, SegmentEvaluator
├── metrics/        # ClassificationMetrics (confusion matrix)
├── cfgs/           # COCO (80 classes), ImageNet (1000 classes)
├── viz/            # DetectVisualizer (draw boxes/labels/scores)
└── utils/          # Logger (get_logger), Profiler (Profile context manager)
```

### Registry Mechanism

Five global registries in `modelflow/core/registry.py`:

```python
from modelflow.core import BACKENDS, PROCESSORS, DATASETS, METRICS, EVALUATORS

# Decorator pattern: register anything that implements the base ABC
@BACKENDS.register("my_backend")
class MyBackend(BaseBackend): ...

# Then build at runtime via the registry
backend = BACKENDS.build("my_backend", model_path="...", class_list=[])
```

The registry pattern allows adding new backends, tasks, datasets, metrics, or evaluators without modifying core framework code.

### Key Contracts

- **Backend contract**: takes `np.ndarray` (NCHW float32 preprocessed), returns `List[np.ndarray]` (raw model outputs). No image processing inside backends.
- **Pipeline contract**: `pipeline(image)` runs preprocess → infer → postprocess end-to-end. `pipeline.infer(tensor)` skips pre/post for direct backend access.
- **Evaluator contract**: `evaluator.run()` iterates dataset, accumulates predictions, returns `Dict[str, float]` of metrics. Detection/segment evaluators delegate mAP to DataFlow-CV (gracefully fall back if not installed).

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

#### Evaluator DataFlow-CV Bridge

Detection and segmentation evaluators delegate mAP to `dataflow.evaluate.DetectionEvaluator`:

```python
# DetectEvaluator internally:
# 1. Runs inference loop, collects predictions in COCO format
# 2. Saves pred JSON (optional, via save_pred_json)
# 3. If gt_json and DataFlow-CV available:
#    gt_coco = COCO(gt_json)
#    dt_coco = gt_coco.loadRes(pred_json)
#    evaluator = DetectionEvaluator()
#    result = evaluator.evaluate(gt_coco, dt_coco)
# 4. Fallback: return {"num_predictions": N}
```

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

Package version: `modelflow/__init__.py` has `__version__ = "0.1.0"`; the CHANGELOG records project history from v0.3.0.

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

# Visualization
pip install opencv-python      # Required for viz module
```

### Test Suite (96 tests + export tests)

All tests use pytest without external fixtures. No model files are required — backends test import/init only (not actual inference).

#### Test Structure

```
tests/                          # modelflow package tests
├── test_core.py                # Registry, Types, Config, Interfaces, Pipeline
├── test_backends.py            # OnnxBackend, TensorrtBackend, TritonBackend init
├── test_pipelines.py           # Factory functions reject invalid backends
├── test_processors.py          # All 5 processor families + detect ops
├── test_metrics.py             # ClassificationMetrics (confusion matrix)
├── test_datasets.py            # COCODetection, ClassifyDir, COCOSegment
└── test_cfgs.py                # COCO 80 classes, ImageNet 1000 classes

export/tests/                   # export module tests
├── test_export.py              # Preprocessing, exporters, validation
└── test_engine.py              # TRT calibrators, FP16 build, Triton config
```

Test coverage includes:
- **Registry**: register, get, build, list, contains, dedup warnings
- **Types**: enum values, ModelInfo dataclass
- **Config**: defaults, serialization (to_dict/from_dict/from_json)
- **Interfaces**: ABC subclassing, Pipeline composition, warmup cascade
- **Processors**: shape/type/value constraints across all 5 task families; NMS correctness; empty-detection edge cases; CLIP similarity; colormap
- **Metrics**: confusion matrix, perfect/imperfect accuracy, reset
- **Datasets**: empty directories, getitem returns expected structure

### Known Gotchas

1. **Backend must not do image processing**: Pre-processing (resize, normalize, letterbox) must happen in Preprocessor, not in Backend. Backend's sole job is `np.ndarray` in → `List[np.ndarray]` out.
2. **YOLO v5 vs v8/v11 output shape**: The two versions have different output tensor shapes (`(1, 25200, 85)` vs `(1, 84, 8400)`). The `model_version` parameter controls which decode path is used. Mixing them up silently produces garbage results.
3. **Empty detection handling**: Postprocessor must handle the case where no detections pass the confidence threshold. Return empty arrays with correct dtypes (shape `(0, 4)` for boxes, `(0,)` for scores/class_ids), not None.
4. **DataFlow-CV optional dependency**: Detection and segmentation evaluators gracefully degrade when DataFlow-CV is not installed. Always guard `import dataflow` with try/except — never fail hard.
5. **TensorRT device buffer cleanup**: `TensorrtBackend.__del__` frees CUDA device buffers. If the backend object is not properly garbage-collected (e.g., circular references), GPU memory leaks. Use explicit `del backend` or `with` context patterns.
6. **OnnxRuntime provider fallback**: `OnnxBackend` auto-selects CUDAExecutionProvider when CUDA is available. On systems without CUDA, it silently falls back to CPU. Set `providers` explicitly to force a specific provider.
7. **Triton model name confusion**: In `TritonBackend`, `model_path` doubles as the Triton model name (not an actual file path). This differs from OnnxBackend and TensorrtBackend where model_path is a file path.
8. **CLIP mean/std values differ**: CLIP preprocessor uses different mean/std from standard ImageNet normalization. Using ImageNet normalize on CLIP inputs produces incorrect results.
9. **Export module zero-dependency**: `export/` contains its own copy of letterbox and preprocessing utilities in `export/core/utils.py`. Never import from `modelflow/` inside `export/`.
10. **NMS NumPy implementation**: The NMS in `modelflow/processors/detect/ops.py` is a pure NumPy implementation. It does not use `torchvision.ops.nms` or any CUDA acceleration. Large batches may be slow.
11. **Triton gRPC vs HTTP**: `TritonBackend` defaults to gRPC (port 8001). HTTP uses port 8000. The `protocol` parameter controls this. Mixing them up silently fails with connection refused.
12. **Segment proto mask dimensions**: The prototype mask output is always `(1, 32, 160, 160)` regardless of input size. The mask decode logic (`process_mask`) must account for this fixed spatial dimension.

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

The Co-Authored-By line is optional and can be omitted.

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
4. **DataFlow-CV**: Optional dependency for detection/segmentation mAP. If absent, evaluators return `{"num_predictions": N}`.
5. **C++ module**: `cpp/` is independently extractable (own CMakeLists.txt). Precision must align between Python and C++ backends.
6. **Legacy samples**: `samples/yolov5/torch_util.py`, `samples/yolov8/torch_util.py`, `samples/yolov8_seg/torch_util.py` are pre-modelflow inference scripts retained for reference. Use `samples/infer.py` instead.

## References

- `README.md` — project overview, install guide, and Python API examples
- `specs/SDD_AGENT.md` — **SDD Agent development methodology (start here for development)**
- `specs/index.md` — architecture specification index
- `specs/modules/` — detailed module design specs
- `specs/export/` — ONNX/TensorRT/Triton knowledge layer
- `export/README.md` — export module documentation
- `modelflow/README.md` — modelflow package documentation
- `llms/clip_samples/README.md` and `llms/openclip_samples/README.md` — CLIP evaluation docs