# Export Module

> Convert PyTorch models into production-ready inference artifacts through a three-stage pipeline.

<br>

## 1. Pipeline

```
 PyTorch (.pt)
      │
      ▼
  ┌──────────┐
  │  L1 ONNX  │    PT → ONNX
  └─────┬─────┘    cross-platform · CPU / GPU
        │
        │  .onnx
        ▼
  ┌──────────┐
  │ L2 TensorRT│   ONNX → Engine
  └─────┬─────┘   FP16 / INT8 · GPU only
        │
        │  .engine
        ▼
  ┌──────────┐
  │ L3 Triton │    Config + Repo
  └───────────┘    production serving
```

> **Each stage is independent.** Stop at L1 for ONNX Runtime. Go L2 for GPU acceleration. Reach L3 for production serving. You can also go straight from L1 to L3 without L2.

---

## 2. L1 · ONNX Export

Export pretrained models to ONNX for cross-platform inference.

### 2.1 Classification

Covers torchvision model families: `ResNet` · `EfficientNet` · `MobileNet` · `ShuffleNet` · `SqueezeNet` · `MNASNet` · `DenseNet` · `VGG` · `ConvNeXt` · `ViT`

```bash
python3 -m export.onnx.convert \
    --model efficientnet_b0 \
    --save models/runtime/efficientnet_b0.onnx \
    --img-size 224
```

### 2.2 Detection · Segmentation · Pose

YOLOv8 / YOLO11 and compatible variants — auto-inferred task type from model name suffix.

```bash
python3 -m export.onnx.ultralytics yolov8s       --save models/runtime/yolov8s.onnx
python3 -m export.onnx.ultralytics yolov8s-seg   --save models/runtime/yolov8s-seg.onnx
python3 -m export.onnx.ultralytics yolov8s-cls   --save models/runtime/yolov8s-cls.onnx
python3 -m export.onnx.ultralytics yolov8s-pose  --save models/runtime/yolov8s-pose.onnx
```

### 2.3 Graph Simplification

`torch.onnx.export` produces redundant graph nodes. `onnx-simplifier` cleans them up — **lossless** optimization, typically 5–15% smaller graph.

| Optimization | Effect |
|---|---|
| Identity removal | `A → Identity → B` ⟹ `A → B` |
| Transpose / Reshape merge | Cancelling pairs collapsed |
| Cast elimination | No-op casts (e.g. `fp32 → fp32`) removed |
| Constant folding | All-constant subgraphs pre-computed |

```bash
python3 -m onnxsim model.onnx model_simplified.onnx      # CLI
```

```python
from export.onnx import optimize_onnx
optimize_onnx("model.onnx", "model_simplified.onnx")     # Python API
```

> If `onnx-simplifier` is missing, `optimize_onnx` prints an install hint and returns the original path — **never fails hard**.

### 2.4 Automatic Validation

Every export runs two checks automatically:

| Check | Method | Tolerance |
|---|---|---|
| ONNX validity | `onnx.checker.check_model()` | must pass |
| PT vs ONNX | `np.allclose` on same random input | `rtol=1e-3, atol=1e-5` |

---

## 3. L2 · TensorRT Build

Build GPU-optimized engines from ONNX models.

### 3.1 FP16 Engine

Two build paths — `trtexec` by default, Python API as fallback.

```bash
# trtexec (no Python TensorRT needed)
python3 -m export.tensorrt.build_fp16 \
    --onnx models/runtime/yolov8s.onnx \
    --save models/tensorrt/yolov8s_fp16.engine

# Python API
python3 -m export.tensorrt.build_fp16 \
    --onnx models/runtime/yolov8s.onnx \
    --save models/tensorrt/yolov8s_fp16.engine \
    --no-trtexec
```

| | FP16 |
|---|---|
| Precision loss | ~0% |
| Speedup | 1.5–2× |
| Calibration data | none needed |
| GPU requirement | Turing+ (FP16 capable) |

### 3.2 INT8 Engine

Requires **50–100 calibration images**. Two calibrator backends for different environments.

```bash
# Step 1 — generate calibration data
python3 export/tensorrt/scripts/generate_calib_cache_for_coco.py \
    --input_dir path/to/coco_val_subset \
    --output_dir calib_coco_dst

# Step 2a — build (PyTorch calibrator)
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir calib_coco_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640

# Step 2b — build (PyCUDA calibrator, for Jetson / embedded)
python3 -m export.tensorrt.build_int8_pycuda \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir calib_coco_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640
```

| Calibrator | Depends on | Best for |
|---|---|---|
| **TorchCalibrator** | `torch` + `tensorrt` | RTX servers, dev workstations |
| **PyCudaCalibrator** | `pycuda` + `tensorrt` | Jetson, embedded, Docker minimal |

| | INT8 |
|---|---|
| Precision loss | ~0.5–1% mAP |
| Speedup | 2–3× |
| Calibration data | 50–100 images required |
| GPU requirement | Turing+ (INT8 capable) |

### 3.3 Calibration Data Scripts

```bash
# ImageNet classification
python3 export/tensorrt/scripts/generate_calib_cache_for_imagenet.py \
    --input_dir path/to/imagenet_val_subset \
    --output_dir calib_imagenet_dst \
    --crop_size 224

# COCO detection / segmentation
python3 export/tensorrt/scripts/generate_calib_cache_for_coco.py \
    --input_dir path/to/coco_val_subset \
    --output_dir calib_coco_dst \
    --input_size 640

# Random subset sampler (helper)
python3 export/tensorrt/scripts/random_copy_images.py \
    /path/to/source_dataset /path/to/calib_subset 100
```

---

## 4. L3 · Triton Deploy

Generate a Triton model repository with task-aware I/O dimensions.

### 4.1 Config Generation

Auto-detects input/output dims for: `classify` · `detect` · `segment` · `pose`

```bash
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt \
    --task detect \
    --save ./models/triton/
```

### 4.2 Model Repository

```python
from export.triton import ModelRepoBuilder, TritonConfigGenerator

builder = ModelRepoBuilder("models/triton/")
builder.deploy(
    model_name="Detect_COCO_YOLOv8s_TRT",
    model_file="models/tensorrt/yolov8s_fp16.engine",
    backend="tensorrt",
    version=1,
)

gen = TritonConfigGenerator("Detect_COCO_YOLOv8s_TRT", backend="tensorrt", task="detect")
gen.save("models/triton/")
```

Expected directory layout after generation:

```
models/triton/
└── Detect_COCO_YOLOv8s_TRT/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

### 4.3 Naming Convention

```
{Task}_{Dataset}_{Architecture}_{Backend}

Detect_COCO_YOLOv8s_ONNX        Classify_ImageNet_EfficientNetB0_TRT
Detect_COCO_YOLOv8s_TRT         Segment_COCO_YOLOv8sSeg_TRT
```

### 4.4 Launch Server

```bash
docker run --gpus=all -it \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models/triton:/models \
    nvcr.io/nvidia/tritonserver:24.06-py3 \
    tritonserver --model-repository=/models
```

| Port | Protocol | Purpose |
|---|---|---|
| 8000 | HTTP | REST inference |
| 8001 | gRPC | High-perf inference |
| 8002 | HTTP | Prometheus metrics |

---

## 5. Package Structure

```
export/
├── _base.py                    # BaseExporter ABC
├── _validation.py              # ONNX checker + PT comparison
├── _utils.py                   # Preprocessing pipeline (NumPy)
│
├── onnx/                       # L1  —  PT → ONNX
│   ├── convert.py              #   torchvision (30+ architectures)
│   ├── ultralytics.py          #   YOLOv8 / YOLO11 (detect · seg · cls · pose)
│   └── optimize.py             #   onnx-simplifier wrapper
│
├── tensorrt/                   # L2  —  ONNX → TensorRT
│   ├── build_fp16.py           #   FP16 engine (trtexec + Python API)
│   ├── build_int8.py           #   INT8 engine (PyTorch calibrator)
│   ├── build_int8_pycuda.py    #   INT8 engine (PyCUDA calibrator)
│   ├── calibrator.py           #   Base + Torch + PyCuda calibrators
│   └── scripts/
│       ├── generate_calib_cache_for_coco.py
│       ├── generate_calib_cache_for_imagenet.py
│       └── random_copy_images.py
│
├── triton/                     # L3  —  Triton deployment
│   ├── config_generator.py     #   config.pbtxt auto-generation
│   └── model_repo.py           #   model repository builder
│
└── tests/
    ├── test_export.py
    └── test_engine.py
```

---

## 6. Python API

```python
# ── L1 ──────────────────────────────────────────────

from export.onnx import TorchvisionExporter
TorchvisionExporter("efficientnet_b0").export_onnx("model.onnx", img_size=224)

from export.onnx import UltralyticsExporter
UltralyticsExporter("yolov8s").export_onnx("yolov8s.onnx")

# ── L2 ──────────────────────────────────────────────

from export.tensorrt import build_fp16_engine
build_fp16_engine("model.onnx", "model_fp16.engine")

from export.tensorrt import build_int8_engine_torch
build_int8_engine_torch(
    onnx_path="model.onnx",
    calib_dir="calib_data",
    output_path="model_int8.engine",
    input_shape=(1, 3, 640, 640),
)

# ── L3 ──────────────────────────────────────────────

from export.triton import TritonConfigGenerator, ModelRepoBuilder

TritonConfigGenerator("Detect_COCO_YOLOv8s_TRT", backend="tensorrt", task="detect") \
    .save("models/triton/")

ModelRepoBuilder("models/triton/") \
    .deploy("Detect_COCO_YOLOv8s_TRT", "yolov8s_fp16.engine")
```

