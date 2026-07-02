# Triton Deployment Configuration Specification

> **Status:** Implemented
> **Version:** 0.1
> **Prerequisite reading:** [`specs/export/onnx_export.md`](onnx_export.md), [`specs/export/tensorrt_conversion.md`](tensorrt_conversion.md)

## 1. Triton Model Repository

Triton Inference Server uses a specific directory structure to organize models, called the **Model Repository**.

### 1.1 Standard Structure

```
models/triton/
└── <model_name>/                    # Model name (unique identifier)
    ├── config.pbtxt                 # Model configuration (text format)
    └── <version>/                   # Version number (positive integer)
        ├── model.onnx               # ONNX model file
        └── model.plan               # Or TensorRT engine file (choose one)
```

**Version management:**
- Each model can have multiple versions (1/, 2/, 3/, ...)
- Triton loads the latest version (largest number) by default
- Controlled via the `version_policy` setting in `config.pbtxt`

### 1.2 Model Naming Convention

```
<Task>_<Dataset>_<ModelArch>_<Backend>

Examples:
- Detect_COCO_YOLOv8s_ONNX
- Classify_ImageNet_EfficientNetB0_TRT
- Segment_COCO_YOLOv8sSeg_ONNX
```

| Field | Allowed Values | Description |
|------|-------|------|
| Task | Detect, Classify, Segment, Pose, SemanticSeg | Task type (SemanticSeg not yet supported by config generator) |
| Dataset | COCO, ImageNet, Custom | Training dataset |
| Model Architecture | YOLOv8s, EfficientNetB0, ResNet50 | Model name |
| Backend | ONNX, TRT | Inference backend |

## 2. config.pbtxt Configuration

### 2.1 ONNX Backend Configuration

```
name: "Detect_COCO_YOLOv8s_ONNX"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [84, 8400]
  }
]
```

### 2.2 TensorRT Backend Configuration

```
name: "Detect_COCO_YOLOv8s_TRT"
platform: "tensorrt_plan"
max_batch_size: 0

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [84, 8400]
  }
]
```

### 2.3 Dynamic Batch Configuration

When dynamic batching needs to be enabled:

```
name: "Detect_COCO_YOLOv8s_TRT"
platform: "tensorrt_plan"
max_batch_size: 8

dynamic_batching {
  preferred_batch_size: [1, 4, 8]
  max_queue_delay_microseconds: 100
}

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [84, 8400]
  }
]
```

**Note:** Enabling dynamic batching requires the ONNX/TensorRT engine to support `max_batch_size`. When building the TensorRT engine, you must configure the corresponding dynamic shape parameters.

### 2.4 Parameter Reference

| Parameter | Description | Required |
|--------|------|:----:|
| `name` | Model name, must match the directory name | Yes |
| `platform` | Backend type: `onnxruntime_onnx` or `tensorrt_plan` | Yes |
| `max_batch_size` | Maximum batch size (0 = no batch dimension, i.e., N in NCHW is fixed within the engine) | Yes |
| `input` | Input definition (name, data_type, dims) | Yes |
| `output` | Output definition | Yes |
| `dynamic_batching` | Dynamic batching policy | Optional |
| `instance_group` | Multi-instance configuration | Optional |
| `version_policy` | Version management policy | Optional |

## 3. ONNX vs TensorRT Backend Comparison

| Comparison | ONNX Runtime Backend | TensorRT Backend |
|--------|:----------------:|:-------------:|
| Model file | `model.onnx` | `model.plan` |
| Inference performance | Baseline | Better (FP16/INT8) |
| Build steps | ONNX export only | ONNX export + TensorRT build required |
| Flexibility | High | Limited by engine shape |
| Load speed | Fast | Relatively slow (requires deserialization) |
| Use case | Quick deployment, newer GPU with FP16 support | Extreme optimization, production |

## 4. Starting Triton Server

Triton Server is started via Docker with the model repository mounted at `/models`. The server exposes three ports: 8000 (HTTP), 8001 (gRPC), 8002 (Prometheus metrics).

**Port Reference:**

| Port | Protocol | Purpose |
|------|------|------|
| 8000 | HTTP | RESTful API inference |
| 8001 | gRPC | gRPC inference (high performance) |
| 8002 | HTTP | Prometheus metrics |

## 5. Automatic Configuration File Generation

The model repository structure and `config.pbtxt` should be generated automatically by scripts to avoid errors from manual authoring.

```bash
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt \
    --task detect \
    --save models/triton/

# Generated structure:
# models/triton/Detect_COCO_YOLOv8s_TRT/
# ├── config.pbtxt
# └── 1/
#     └── model.plan
```

### 5.1 Generation Rules

| Parameter | Description | Affects |
|------|------|------|
| `--model-name` | Model name | Directory name + config.name |
| `--backend` | onnxruntime / tensorrt (alias: trt) | platform + model file extension |
| `--task` | detect / classify / segment | Input/output dims |
| `--version` | Version number | Subdirectory name |
| `--max-batch` | Maximum batch size | max_batch_size + dynamic_batching |
| `--save` | Output path | Repository root directory |

## 6. Verification

| Check | Method |
|--------|------|
| Config file format | `tritonserver --model-repository=/models --strict-model-config=false` startup log |
| Model loading | Triton log should show `"READY"` status |
| Inference verification | Send requests using Triton Python Client and compare outputs |
