# TensorRT Conversion Specification

> **Status:** Implemented
> **Version:** 0.1
> **Prerequisite Reading:** [`specs/export/onnx_export.md`](onnx_export.md) (ONNX Export Specification), TensorRT fundamentals

> **Note:** This module is implemented based on the TensorRT 10.x API. `.engine` files built with TensorRT 10.x cannot be loaded in TensorRT 8.x environments. When deploying, ensure the Triton Server version is >= 24.06 (bundles TRT 10.x).

## 1. How TensorRT Works

TensorRT is NVIDIA's high-performance deep learning inference optimizer. It converts a trained model (ONNX) into an inference engine through **graph optimization** and **kernel auto-tuning**.

```
ONNX Computation Graph
    │
    ▼
┌──────────────────────────────┐
│      1. Graph Parse           │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│   2. Graph Optimization       │
│   ├── Layer Fusion            │
│   ├── Constant Folding        │
│   ├── Dead Branch Elimination │
│   └── Precision Calibration   │
│       (INT8 Quantization)     │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  3. Kernel Auto-Tuning        │
│   └── Selects optimal CUDA    │
│       kernel for each layer   │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│    4. Serialize → .engine     │
└──────────────────────────────┘
```

**Key optimization techniques:**

| Optimization | Description | Effect |
|------|------|------|
| **Layer Fusion** | Merges Conv+BN+ReLU into a single kernel | Reduces memory bandwidth consumption and kernel launch overhead |
| **Precision Calibration** | FP32 → INT8, finds optimal quantization threshold via KL divergence | 2–3× speedup |
| **Kernel Auto-Tuning** | Tries multiple CUDA kernels per operator, picks the best | Optimal hardware utilization |
| **Memory Reuse** | Analyzes tensor lifetimes, reuses memory | Reduces peak memory usage |

## 2. FP16 Conversion

### 2.1 Principles

FP16 (half-precision floating point) truncates FP32's 32-bit weights and activations down to 16 bits:

| Format | Sign Bit | Exponent Bits | Mantissa Bits | Representable Range |
|------|-------|-------|-------|---------|
| FP32 | 1 | 8 | 23 | ~3.4×10³⁸ |
| FP16 | 1 | 5 | 10 | ~6.6×10⁴ |

**Precision loss:** Typically ~0% (nearly imperceptible for vision models)

**Speedup:** 1.5–2× (depends on model and GPU architecture)

### 2.2 Build Methods

**Method A: trtexec CLI (recommended)**

```bash
trtexec --onnx=model.onnx \
        --saveEngine=model_fp16.engine \
        --workspace=4096 \
        --fp16
```

Suitable for batch conversion tasks. The output can be used directly for inference and Triton deployment.

**Method B: Python API wrapper**

For scenarios requiring integration into automated pipelines, wrap trtexec calls or use the TensorRT Python API.

### 2.3 Shape Configuration

For fixed-size models (e.g., 640×640 detection), use static shapes:

```
--minShapes=input:1x3x640x640 \
--optShapes=input:1x3x640x640 \
--maxShapes=input:1x3x640x640
```

See Section 4 below for dynamic shape details.

### 2.4 Applicable Scenarios

| Scenario | Recommendation |
|------|------|
| Tesla T4 / RTX 30/40 series GPU | ✅ Enable by default |
| Extremely high precision required (medical, etc.) | Validate FP16 vs. FP32 difference first |
| Edge devices (Jetson) | ✅ Recommended |
| GPU does not support FP16 (older architectures) | ❌ Use FP32 |

## 3. INT8 Conversion

### 3.1 Principles

INT8 quantization maps FP32 values to the 8-bit integer range `[-128, 127]`:

```
FP32 value ──▶ multiply by scale factor ──▶ round ──▶ INT8
INT8 value  ──▶ multiply by dequantization factor ──▶ FP32 (approximate)
```

**Calibration** is the core of INT8 quantization: run FP32 inference on a small dataset, collect the activation distribution of each layer, and use **KL divergence** to find the optimal quantization threshold that minimizes information loss between pre- and post-quantization.

```
Calibration Dataset (50–100 images)
    │
    ▼
FP32 Inference ──▶ Collect Activation Histograms ──▶ KL Divergence Optimization ──▶ Per-Layer Quantization Threshold
                                                                                      │
                                                                                      ▼
                                                                              INT8 Engine Build
```

### 3.2 Calibration Dataset Requirements

| Model Type | Recommended Dataset | Calibration Images | Preprocessing |
|---------|-----------|-----------|-----------|
| Classification (ImageNet) | ImageNet validation subset | 50–100 | Resize(256) + CenterCrop(224) |
| Detection (COCO) | COCO val2017 subset | 50–100 | LetterBox(640) + /255 |
| Segmentation | COCO val2017 subset | 50–100 | Same as detection preprocessing |

**Calibration data preparation pipeline:**

```
Raw Images ──▶ Preprocessing ──▶ Save as .bin (float32 raw binary)
```

```bash
# Classification model calibration data
python3 export/tensorrt/scripts/generate_calib_cache_for_imagenet.py \
    --input_dir export/cal_imagenet_src \
    --output_dir export/cal_imagenet_dst \
    --crop_size 224

# Detection model calibration data
python3 export/tensorrt/scripts/generate_calib_cache_for_coco.py \
    --input_dir export/cal_coco_src \
    --output_dir export/cal_coco_dst \
    --input_size 640
```

### 3.3 Dual Calibrator Strategy

Two calibrator implementations are provided, depending on the deployment environment's dependencies:

| Feature | PyTorch Calibrator | PyCUDA Calibrator |
|------|:------------------:|:-----------------:|
| **Dependency** | `torch` + `tensorrt` | `pycuda` + `tensorrt` |
| **Startup** | Loads Torch library (slightly slower) | Binds CUDA directly (fast) |
| **Memory Overhead** | ~2 GB or more | <100 MB |
| **Data Copy** | `torch.pin_memory` + `cuda.copy_` | `pycuda.pagelocked_empty` + `memcpy_htod` |
| **Recommended Device** | RTX servers, dev machines | Jetson, embedded, lightweight Docker images |
| **Ease of Debugging** | High (Torch ecosystem) | Low (requires pycuda knowledge) |

**Data self-healing:** Both calibrators implement the following safeguards:
- Auto-detect and skip `.bin` files with mismatched sizes
- Repair NaN/Inf data (`nan_to_num`)
- Per-file exception handling (corruption of a single file does not halt the entire build)

### 3.4 Build Methods

```bash
# PyTorch environment (recommended for dev machines)
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir export/cal_coco_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640 \
    --workspace 4

# PyCUDA environment (recommended for Jetson / embedded)
python3 -m export.tensorrt.build_int8_pycuda \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir export/cal_coco_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640
```

### 3.5 Expected Precision Loss

| Model Type | FP32 → FP16 Loss | FP32 → INT8 Loss |
|---------|:----------------:|:----------------:|
| Classification (ResNet) | < 0.1% | < 0.5% |
| Detection (YOLOv8) | < 0.1% | ~0.5–1% mAP |
| Segmentation | < 0.1% | < 1% mAP |

**Note:** Actual precision loss varies by model and calibration dataset. Always verify precision after each conversion.

## 4. Dynamic Shape Handling

### 4.1 When Dynamic Shapes Are Needed

| Scenario | Recommendation |
|------|------|
| Fixed input size at inference (e.g., YOLOv8 640×640) | ✅ Use static shape |
| Variable input size at inference (e.g., classification models with different resolutions) | Use dynamic shape |
| Batch inference required | Use dynamic batch |

### 4.2 Configuration

```bash
# Dynamic batch example
--minShapes=input:1x3x640x640 \
--optShapes=input:8x3x640x640 \
--maxShapes=input:16x3x640x640

# Dynamic width/height example
--minShapes=input:1x3x224x224 \
--optShapes=input:1x3x640x640 \
--maxShapes=input:1x3x1280x1280
```

**Note:**
- Dynamic shape engines are typically slower than fixed-shape engines
- INT8 calibration is more complex with dynamic shapes
- Triton supports dynamic shape engines

## 5. Quantization Strategy Selection

### 5.1 Decision Tree

```
Can the model run on GPU?
├── No ──▶ ONNX Runtime (CPU) — no need for TensorRT
└── Yes ──▶ Need maximum performance?
    ├── No ──▶ ONNX Runtime (GPU) — or FP16
    └── Yes ──▶ How tolerant are you of precision loss?
        ├── Relaxed ──▶ INT8 (2–3× speedup)
        └── Strict ──▶ FP16 (1.5–2× speedup)
```

### 5.2 Quick Comparison

| Factor | FP16 | INT8 |
|------|------|------|
| Speedup | 1.5–2× | 2–3× |
| Precision Risk | Very low | Low–medium |
| Calibration Data | Not needed | 50–100 images needed |
| Build Time | A few minutes | Tens of minutes (calibration overhead) |
| Applicable Scenarios | Default choice | Maximum performance, edge devices |
| Hardware Requirement | FP16 support | INT8 support (Turing+) |

## 6. Verification

| Check Item | Method | Criterion |
|--------|------|------|
| FP32 vs. FP16 precision | Compare outputs on the same input | Precision loss < 0.1% |
| FP32 vs. INT8 precision | Compare outputs on the same input | Precision loss < 1% (mAP) |
| Engine loading | `trtexec --loadEngine=model.engine` | Loads successfully |
| Inference stability | Run inference 100+ times consecutively | Output stable, no NaN |
