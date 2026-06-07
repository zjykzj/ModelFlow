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

All tests use pytest without external fixtures. No model files are required — backends test import/init only (not actual inference). Test coverage includes:

- **Registry**: register, get, build, list, contains, dedup warnings
- **Types**: enum values, ModelInfo dataclass
- **Config**: defaults, serialization (to_dict/from_dict/from_json)
- **Interfaces**: ABC subclassing, Pipeline composition, warmup cascade
- **Processors**: shape/type/value constraints across all 5 task families; NMS correctness; empty-detection edge cases; CLIP similarity; colormap
- **Metrics**: confusion matrix, perfect/imperfect accuracy, reset
- **Datasets**: empty directories, getitem returns expected structure

### Git Conventions

- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `perf:`, `test:`
- Large model files (`.onnx`, `.engine`, `.plan`, `.trt`) are gitignored — do not commit them
- Calibration data directories (`export/cal_coco_src/`, `export/cal_imagenet_src/`) are gitignored
- Output directories (`output/`, `outputs/`, `models/`) are gitignored

### Important Considerations

1. **Model paths**: Store models in `models/{runtime,tensorrt,triton}/` matching the backend.
2. **Dynamic shapes**: TensorRT engines require fixed input shapes during conversion; use `--input_shape`.
3. **Calibration data**: INT8 quantization requires calibration datasets (COCO or ImageNet subsets).
4. **DataFlow-CV**: Optional dependency for detection/segmentation mAP. If absent, evaluators return `{"num_predictions": N}`.
5. **C++ module**: `cpp/` is independently extractable (own CMakeLists.txt). Precision must align between Python and C++ backends.
6. **Legacy samples**: `samples/yolov5/torch_util.py`, `samples/yolov8/torch_util.py`, `samples/yolov8_seg/torch_util.py` are pre-modelflow inference scripts retained for reference. Use `samples/infer.py` instead.

## References

- `README.md` — project overview, install guide, and Python API examples
- `specs/index.md` — architecture specification index
- `specs/modules/` — detailed module design specs
- `specs/export/` — ONNX/TensorRT/Triton knowledge layer
- `export/README.md` — export module documentation
- `modelflow/README.md` — modelflow package documentation
- `llms/clip_samples/README.md` and `llms/openclip_samples/README.md` — CLIP evaluation docs