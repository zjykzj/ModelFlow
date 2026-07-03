# Model Export Module Specification

> **Version:** 0.2 | **Last Updated:** 2026-07-03
> **Status:** Implemented
> **Dependencies:** `spec_architecture.md` (independent module design principles), `specs/export/` (knowledge layer)

## 1. Module Positioning

`export/` is an **independent module** with zero dependencies on `modelflow/` or any other ModelFlow module. Its responsibility is to convert PyTorch models to ONNX, further convert them to TensorRT engines, and generate Triton model repository configurations.

```
PyTorch (.pt) ──▶ ONNX (.onnx) ──▶ TensorRT (.engine) FP16 / INT8
                                      │
                                      ▼
                               Triton model repository (config.pbtxt + model)
```

**Knowledge layer reference:** For format understanding, conversion principles, model differences, and selection rationale, see the [`specs/export/`](../export/index.md) document series.

### 1.1 Design Constraints

| # | Constraint | Description | Violation Example |
|---|------|------|---------|
| 1 | **Zero external dependencies** | Code under `export/` must not import from `modelflow/` or any other ModelFlow module | ❌ `from modelflow.processors import ...` |
| 2 | **Self-contained preprocessing** | Preprocessing logic in calibration data preparation scripts (`tensorrt/scripts/`) must be self-implemented, or use utilities provided at the `export/` root (`_utils.py`) | |
| 3 | **Dependency version constraints** | Allowed third-party dependencies: `torch`, `onnx`, `onnxruntime`, `tensorrt` (>=10.0), `pycuda` (optional), `ultralytics` (optional) | |
| 4 | **Lazy loading of optional components** | Optional dependencies such as `tensorrt`, `pycuda`, `ultralytics` must not be imported at module level; they must be imported on-demand at the point of use, with `ImportError` caught to provide installation guidance | ❌ Module-level `import tensorrt as trt` blocks submodules that don't depend on tensorrt (e.g., `build_fp16` cannot run in a non-TRT environment) |

**Background:** This constraint ensures that the `export/` module can be developed and tested independently, without being blocked by changes in other project modules. When preprocessing logic already exists in `modelflow/` or project root `core/` and is needed here, it should be **copied or refactored** into `export/_utils.py` rather than imported from the original location.

## 2. Directory Structure

`export/` is organized by the three stages of the export pipeline, with common infrastructure placed directly at the module root (prefixed with `_` to indicate internal shared code).

**Structure description:**

| Submodule | Responsibility | Knowledge Layer Document |
|--------|------|-----------|
| `_base.py` / `_validation.py` / `_utils.py` | Common infrastructure: exporter base class, ONNX validation, preprocessing utilities | — |
| `onnx/` | L1: PT→ONNX export | [`specs/export/onnx_export.md`](../export/onnx_export.md) |
| `tensorrt/` | L2: ONNX→TensorRT engine build (including calibration scripts) | [`specs/export/tensorrt_conversion.md`](../export/tensorrt_conversion.md) |
| `triton/` | L3: Triton config generation and model repository deployment | [`specs/export/triton_deployment.md`](../export/triton_deployment.md) |
| `tests/` | Export tests and engine tests | — |

## 3. Export Pipeline

### 3.1 Pipeline Stages

```
Stage 1: PT → ONNX
    ├── torchvision models: torch.onnx.export manual wrapper
    └── Ultralytics models: YOLO().export(format="onnx")
            │
            ▼
Stage 2: ONNX → TensorRT  (optional)
    ├── FP16: fast, negligible precision loss, no calibration data needed
    └── INT8: requires calibration data and dataset preprocessing
            │
            ▼
Stage 3: Triton configuration   (optional)
    └── Generate config.pbtxt + model repository structure
```

### 3.2 Export Depth Levels

| Level | Output | Applicable Scenario | Command Reference |
|------|------|---------|---------|
| **L1** | `.onnx` | ONNX Runtime inference | `export/onnx/convert.py` or `export/onnx/ultralytics.py` |
| **L2** | `.onnx` + `.engine` | TensorRT GPU inference | L1 + `export/tensorrt/build_fp16.py` or `build_int8.py` |
| **L3** | `.onnx` + `.engine` + Triton config | Triton Server deployment | L1/L2 + `export/triton/config_generator.py` |

## 4. Task Export Specification

### 4.1 Model Sources and Tasks

| Model Source | Task Type | L1 | L2 | L3 |
|---------|---------|:--:|:--:|:--:|
| torchvision | Classification | ✅ | ✅ | ✅ |
| Ultralytics | Detection | ✅ | ✅ | ✅ |
| Ultralytics | Segmentation | ✅ | ✅ | ✅ |
| Ultralytics | Classification | ✅ | ✅ | ✅ |
| Ultralytics | Pose | ✅ | ✅ | ✅ |

### 4.2 Input/Output Specification

| Task | Input Shape | Input Name | Output Description |
|------|-----------|--------|---------|
| Classification | `(1, 3, 224, 224)` | `image` | logits `(1, num_classes)` |
| Detection | `(1, 3, 640, 640)` | `image` | pred `(1, 84, 8400)` |
| Segmentation | `(1, 3, 640, 640)` | `image` | pred `(1, 116, 8400)` + proto mask |
| Pose | `(1, 3, 640, 640)` | `image` | pred `(1, 56, 8400)` |

## 5. Precision Validation

| Check Item | Method | Criterion |
|--------|------|------|
| ONNX validity | `onnx.checker.check_model()` | No error |
| PT vs ONNX | Compare outputs from the same input | rtol=1e-3, atol=1e-5 |
| FP32 vs FP16 | Compare outputs from the same input | FP16 precision loss < 0.1% |
| FP32 vs INT8 | Compare outputs from the same input | INT8 precision loss < 1% |

## 6. Dependency Contract

### 6.1 Allowed Imports

The `export/` module may import from:

| Source | Used By | Purpose |
|--------|---------|---------|
| `torch` / `torchvision` | `onnx/`, `tensorrt/` | Model loading, tracing, calibration |
| `onnx` / `onnxruntime` | `_validation.py`, `onnx/` | ONNX model validation, inference comparison |
| `tensorrt` | `tensorrt/` | Builder, network definition, engine serialization |
| `numpy` | All submodules | Array operations, calibration data preparation |
| `pycuda` | `tensorrt/build_int8_pycuda.py` | GPU-accelerated INT8 calibration (Jetson) |
| `PIL` / `cv2` | `tensorrt/scripts/` | Calibration data preprocessing |
| `export/_utils.py` | All submodules | Self-contained letterbox/preprocessing (NumPy only) |

### 6.2 Forbidden Imports

The `export/` module must NOT import from:

| Forbidden Source | Reason |
|-----------------|--------|
| `modelflow/*` | **Architecture constraint**: Export module is self-contained. Its preprocessing in `_utils.py` is an independent implementation. |

### 6.3 Import Rationale

Export's `_utils.py` duplicates some preprocessing logic from `modelflow/processors/` (letterbox, resize, normalize). This duplication is **intentional** — it keeps `export/` self-contained and avoids a dependency that would break architecture constraint #1.

## 7. Error Handling Contract

| Error Scenario | Behavior |
|---------------|----------|
| Model file not found (.pt) | Raise `FileNotFoundError` with path |
| ONNX export failed (unsupported op) | Raise `RuntimeError` with op name |
| ONNX model validation failed | Raise `ValueError` with checker error details |
| PT vs ONNX output mismatch | Log warning with max difference; do not abort |
| TensorRT build failed (unsupported layer) | Raise `RuntimeError` with layer info |
| INT8 calibration data insufficient | Log warning; fall back to FP16 if possible |
| Triton config generation — unknown task | Raise `ValueError` with supported task list |
| Dynamic shape suppression failed | Log warning; the engine will only accept fixed shapes |

## 8. See Also

- [`specs/export/`](../export/index.md) — export knowledge layer (format principles, conversion specifications)
- [`spec_architecture.md`](spec_architecture.md) — module architecture, dependency contract
