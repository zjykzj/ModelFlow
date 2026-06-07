# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ModelFlow is a computer vision model deployment toolkit focused on **model evaluation, export, and inference**. The primary goal is to deploy computer vision algorithms (object classification, detection, and instance segmentation) with support for multiple inference backends:

- **ONNX Runtime** (with NumPy or PyTorch pre/post-processing)
- **TensorRT** (FP16 and INT8 quantization support)
- **Triton Inference Server** (with ONNX and TensorRT backends)

Supported models: YOLOv5, YOLOv8, YOLOv8-seg (instance segmentation), EfficientNetB0, CLIP/OpenCLIP.

## Common Commands

### Model Inference

```bash
# YOLOv5 inference with ONNX Runtime (NumPy processing)
python3 samples/yolov5/infer.py models/yolov5s.onnx assets/bus.jpg core/cfgs/coco.yaml

# YOLOv5 inference with TensorRT
python3 samples/yolov5/infer.py models/yolov5s_fp16.engine assets core/cfgs/coco.yaml --backend tensorrt

# YOLOv5 inference with Triton
python3 samples/yolov5/infer.py DET_YOLOv5s_ONNX assets core/cfgs/coco.yaml --backend triton

# Specify backend and processor type
python3 samples/yolov5/infer.py model.onnx image.jpg data.yaml --backend onnxruntime --processor numpy
```

### Model Export and Conversion

```bash
# PyTorch to ONNX export (using Ultralytics export script)
python3 export.py --weights yolov5s.pt --include onnx --opset 12

# ONNX to TensorRT (FP16)
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp16.engine --workspace=4096 --fp16

# ONNX to TensorRT (INT8 calibration)
python3 export/safe_int8_build_by_torch.py --onnx export/yolov5s.onnx --calib_dir export/coco_calib_dst --output yolov5s_int8.engine --input_shape 1 3 640 640

# Generate calibration cache for ImageNet
python3 export/scripts/generate_calib_cache_for_imagenet.py --input_dir export/cal_imagenet_src --output_dir export/cal_imagenet_dst

# Test TensorRT engine
trtexec --loadEngine=yolov5s_slim_fp16.engine --iterations=100
```

### Model Evaluation and Benchmarking

```bash
# Run evaluation benchmarks from eval/ directory
python3 eval/runtime/bench_yolov5_onnx_npy.py
python3 eval/trt/bench_yolov5_tensorrt_npy.py
python3 eval/triton/bench_yolov5_triton_npy.py
```

### Triton Server Deployment

```bash
# Start Triton server with model repository
docker run --gpus=all -it -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=./models/triton/

# For TensorRT models in Triton (inside container)
/usr/src/tensorrt/bin/trtexec --onnx=yolov5s.onnx --saveEngine=model.plan --workspace=4096 --fp16
```

### Dataset Preparation

```bash
# Download COCO dataset
bash assets/get_coco.sh

# Download COCO128 dataset
bash assets/get_coco128.sh

# Download ImageNet (requires manual setup)
bash assets/get_imagenet.sh
```

### LLM Evaluations (CLIP/OpenCLIP)

CLIP and OpenCLIP models are evaluated on CIFAR-10 and CIFAR-100 datasets. Installation:

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Evaluation scripts are in `llms/clip_samples/` and `llms/openclip_samples/`.

## Architecture

### Directory Structure

```
ModelFlow/
├── modelflow/             # Python inference/evaluation/visualization core package
│   ├── core/             # Infrastructure: interfaces, registry, types, config
│   ├── cfgs/             # Dataset class configurations (COCO, ImageNet)
│   ├── backends/         # Inference backends (ONNX Runtime, TensorRT, Triton)
│   ├── processors/       # Pre/post-processors by task (classify, detect, segment)
│   ├── pipelines/        # Pre-built InferencePipeline factories
│   ├── datasets/         # Dataset loaders (COCO, classification)
│   ├── evaluators/       # Evaluation orchestrators
│   ├── metrics/          # Local metric implementations
│   ├── viz/              # Visualization (DataFlow-CV bridge + local)
│   └── utils/            # Logger, profiler
├── export/                # Model export and conversion (pt → onnx → tensorrt/triton)
│   ├── core/             # Export infrastructure (base class, validation, preprocessing)
│   ├── onnx/             # PT→ONNX for torchvision and Ultralytics
│   ├── tensorrt/         # TensorRT engine builders (FP16/INT8)
│   ├── triton/           # Triton config generation and model repo management
│   ├── scripts/          # Calibration data preparation utilities
│   └── tests/            # Unit and integration tests
├── models/               # Pre-trained model files
│   ├── runtime/         # ONNX models
│   ├── tensorrt/        # TensorRT engines
│   └── triton/          # Triton model repository
├── core/                 # Legacy inference/evaluation implementations (being migrated)
│   ├── backends/        # Backend implementations (ONNX, TRT, Triton)
│   ├── npy/             # NumPy-based pre/post-processing
│   ├── tch/             # PyTorch-based pre/post-processing
│   ├── cfgs/            # COCO/ImageNet class lists
│   └── utils/           # Logging and utilities
├── eval/                # Evaluation benchmark scripts (legacy)
├── samples/             # Example inference scripts (legacy, being migrated to modelflow)
│   ├── yolov5/
│   ├── yolov8/
│   └── yolov8_seg/
├── llms/                # Language model evaluations (CLIP/OpenCLIP)
└── assets/              # Assets and data scripts
```

### Key Architectural Patterns

1. **Unified Pipeline Pattern**: `modelflow/` provides the `InferencePipeline = Preprocessor + Backend + Postprocessor` abstraction with clear interfaces for all components.

2. **Abstract Backend Interfaces**: `modelflow/backends/` provides `BaseBackend` with implementations for ONNX Runtime, TensorRT, and Triton through `OnnxBackend`, `TensorrtBackend`, `TritonBackend`.

3. **Task-Driven Processors**: `modelflow/processors/` organizes pre/post-processing by task type (classify, detect, segment), with `model_version` parameter for YOLO version differences.

4. **Registry Mechanism**: `modelflow/core/registry.py` provides `Registry` class that enables adding new backends/tasks/datasets without modifying core framework code.

5. **Evaluator Framework**: `modelflow/evaluators/` orchestrates Pipeline + Dataset + Metrics; detection/segmentation metrics delegate to DataFlow-CV.

### Key File Paths

- **Python Inference Package**: `/home/zjykzj/cc/ModelFlow/modelflow/`
- **Export Module**: `/home/zjykzj/cc/ModelFlow/export/`
- **Legacy Backends**: `/home/zjykzj/cc/ModelFlow/core/backends/`
- **Specifications**: `/home/zjykzj/cc/ModelFlow/specs/`
- **Export Utilities**: `/home/zjykzj/cc/ModelFlow/export/`
- **Evaluation Scripts**: `/home/zjykzj/cc/ModelFlow/eval/`

## Development Notes

### Environment Setup

- **No explicit dependency management** – likely uses system-wide Python packages
- **Docker containers** are used for TensorRT and Triton development:
  ```bash
  # TensorRT development container
  docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/zjykzj:/workdir --workdir=/workdir --name ultra ultralytics/yolov5:v7.0 bash

  # Triton server container
  docker run --gpus=all -it -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3 bash
  ```
- **PyCUDA installation** may be required for TensorRT calibration:
  ```bash
  pip3 install pycuda==2024.1.2
  ```
- **CUDA version**: The project targets CUDA 11.x (based on container tags).

### Git Commit Convention

Follows conventional commits with prefixes: `perf`, `feat`, `docs`, `style`, etc.

### Git Ignore Patterns

The `.gitignore` file excludes large files and model artifacts:
- Model files: `*.onnx`, `*.engine`, `*.plan`, `*.trt`
- Output directories: `output/`, `outputs/`, `models/`
- Calibration data directories: `export/cal_coco_src/`, `export/cal_imagenet_src/`
- Python cache and environment directories

Do not commit these files; they are generated during export/inference.

### Important Considerations

1. **Model paths**: Models are stored in `models/` directory with subdirectories per backend (`runtime/`, `tensorrt/`, `triton/`).
2. **Dynamic shapes**: TensorRT engines may require specific input shapes during conversion.
3. **Calibration data**: INT8 quantization requires calibration datasets (COCO, ImageNet subsets).
4. **Triton model repository**: Requires specific directory structure and configuration files.

### Missing Elements

- No `requirements.txt` or `pyproject.toml` for dependency management
- No test suite or CI/CD configuration
- No Dockerfile for reproducible environment

## References

- Main documentation: `README.md`
- Export instructions: `export/README.md`
- Triton deployment: `eval/triton/README.md`
- LLM evaluations: `llms/clip_samples/README.md` and `llms/openclip_samples/README.md`