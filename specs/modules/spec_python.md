# Python Module Specification

> **Version:** 0.5 | **Last Updated:** 2026-07-03
> **Status:** Implemented
> **Dependencies:** `spec_architecture.md` (Pipeline pattern, module relationships), `spec_eval.md` (evaluation module)

## 1. Module Structure

`modelflow/` is organized into backends, processors, and pipelines subpackages. `interfaces.py`, `types.py`, and `config.py` sit at the package root (flattened — no `core/` subpackage).

> **Note:** Evaluation and metrics have been extracted to the standalone `eval/` module. See `specs/modules/spec_eval.md`. Datasets have been extracted to the standalone `data/` module. `modelflow/` is now a **pure inference engine**.

## 2. Core Interfaces

### 2.1 InferencePipeline

```python
class InferencePipeline:
    """
    Inference pipeline = Preprocessor + Backend + Postprocessor

    Complete data flow:
        image (HWC, BGR) → Preprocessor → tensor (NCHW) → Backend → raw (List[ndarray])
            ├── __call__: raw → Postprocessor → StructuredResult (dict)
            └── infer:     returns raw only (for evaluation scenarios, Evaluator handles postprocessing)

    Usage:
        pipeline = InferencePipeline(preprocessor, backend, postprocessor)
        result = pipeline(image, conf_thres=0.25, iou_thres=0.45)  # end-to-end inference
        raw = pipeline.infer(tensor)  # backend-only inference (used inside evaluation loops)
    """
    def __init__(self, preprocessor, backend, postprocessor): ...
    def __call__(self, image, **kwargs) -> Any: ...
    def infer(self, tensor: np.ndarray) -> List[np.ndarray]: ...
    def warmup(self): ...
```

**Pipeline lifecycle:**

```
Evaluator
    │
    ├── Iterate over Dataset (per image)
    │   ├── preprocessor(image)           → tensor
    │   ├── backend(tensor)               → raw List[ndarray]
    │   ├── postprocessor(raw, **kwargs)  → StructuredResult
    │   └── metrics.update(result, ground_truth)
    │
    └── metrics.compute() → Dict[str, float]
```

### 2.2 BaseBackend

The backend's responsibility is **pure tensor inference**, from preprocessed tensor to raw outputs. It needs to know the task type and class list to interpret outputs correctly:

```python
class BaseBackend(ABC):
    """Inference backend. Pure tensor inference, no image processing."""
    def __init__(
        self,
        model_path: str,
        class_list: List[str],
        task_type: Optional[str] = None,   # classify / detect / segment
        half: bool = False,
        device: Optional[str] = None,
        **kwargs
    ): ...

    @abstractmethod
    def __call__(self, input_data: np.ndarray) -> List[np.ndarray]: ...
    def warmup(self): ...
    def get_input_info(self) -> ModelInfo: ...
    def get_output_info(self) -> List[ModelInfo]: ...
    def print_model_info(self): ...
```

| Implementation | Backend | Notes |
|------|------|------|
| `OnnxBackend` | ONNX Runtime | `onnxruntime.InferenceSession` |
| `TensorrtBackend` | TensorRT | `trt.Runtime` + CUDA buffer |
| `TritonBackend` | Triton Server | gRPC/HTTP client |

### 2.3 BasePreprocessor / BasePostprocessor

Preprocessing and postprocessing have different implementations depending on the task type (classify/detect/segment):

```python
class BasePreprocessor(ABC):
    """Preprocessing. Image → network input tensor."""
    @abstractmethod
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray: ...

class BasePostprocessor(ABC):
    """Postprocessing. Raw outputs → structured results."""
    @abstractmethod
    def __call__(self, raw: List[np.ndarray], **kwargs) -> Any: ...
```

### 2.4 BaseDataset

Dataset ABC is defined in the `data/` module (independent of `modelflow/`):

```python
# data/base.py
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, idx) -> Tuple[np.ndarray, Dict]: ...
    @abstractmethod
    def get_gt_json(self) -> str: ...
```

See `data/` module for `build_dataset()` factory and YAML configs.

### 2.5 BaseEvaluator / BaseMetrics

These ABCs are defined in the `eval/` module (not `modelflow/`). See `specs/modules/spec_eval.md`.

## 3. Inference Backends

### 3.1 OnnxBackend

```python
class OnnxBackend(BaseBackend):
    def __init__(self, model_path: str, config: dict = None): ...
    def __call__(self, input_data) -> List[np.ndarray]: ...
```

Auto-selects CUDAExecutionProvider if CUDA available; silently falls back to CPU.

### 3.2 TensorrtBackend

```python
class TensorrtBackend(BaseBackend):
    def __init__(self, engine_path: str, config: dict = None): ...
    def __call__(self, input_data) -> List[np.ndarray]: ...
```

Manages CUDA page-locked host buffers + device buffers; `__del__` frees them.

### 3.3 TritonBackend

```python
class TritonBackend(BaseBackend):
    def __init__(self, model_name: str, config: dict = None): ...
    def __call__(self, input_data) -> List[np.ndarray]: ...
```

`model_path` doubles as Triton model name (not an actual file path). Supports gRPC (default, port 8001) and HTTP (port 8000).

## 4. Processors

Each vision task maps to a pair of Preprocessor + Postprocessor. Each implementation provides both NumPy and PyTorch versions.

### 4.1 Classify

| Component | Implementation | Notes |
|------|------|------|
| `ClassifyPreprocessor` | npy (PIL resize + crop + normalize) / tch (torchvision) | Two modes: resize / crop |
| `ClassifyPostprocessor` | npy/tch shared softmax | Outputs top-1/top-5 |

### 4.2 Detect

Detection preprocessing and postprocessing must handle **YOLO version differences** (YOLOv5 vs YOLOv8/YOLO11). The differences are primarily:
- **Output structure**: v5 dimension layout is `(1, num_dets, 5+nc)`; v8 is `(1, 84, 8400)` transposed
- **NMS logic**: anchor-based (v5) vs anchor-free (v8)
- **Preprocessing**: the auto pad parameter in letterbox differs slightly

Distinguished via the `model_version` parameter to avoid duplicating implementations per version:

```python
class DetectPreprocessor(BasePreprocessor):
    def __init__(self, input_size=640, model_version="v8", half=False): ...
    def __call__(self, image, **kwargs) -> np.ndarray: ...
```

`model_version` distinguishes v5 vs v8/v11 behavior: v5 disables auto_pad; v8/v11 enable it.

| Component | Implementation | Notes |
|------|------|------|
| `DetectPreprocessor` | npy (OpenCV letterbox) / tch (letterbox) | Stride alignment, pad; supports model_version parameter |
| `DetectPostprocessor` | npy NMS / tch NMS | NMS + scale_boxes + clip_boxes; adapts to v5/v8 output format differences |
| `detect/ops.py` | Shared operators | xywh2xyxy, scale_boxes, clip_boxes, nms, etc. |

### 4.3 Segment (Instance Segmentation)

| Component | Implementation | Notes |
|------|------|------|
| `SegmentPreprocessor` | Same as Detect | letterbox |
| `SegmentPostprocessor` | npy / tch | NMS + process_mask + crop_mask + scale_image |

### 4.4 Semantic Segmentation

| Component | Implementation | Notes |
|------|------|------|
| `SemanticSegPreprocessor` | resize + normalize | ImageNet or dataset statistics |
| `SemanticSegPostprocessor` | argmax + colormap | HWC uint8 mask |

### 4.5 Multi-modal

| Component | Implementation | Notes |
|------|------|------|
| `ImagePreprocessor` | CLIP standard preprocessing | resize 224 + center crop + normalize |
| `TextPreprocessor` | CLIP tokenizer | Text → token IDs |
| `Postprocessor` | softmax + ranking | similarity → probability |

## 5. Evaluation & Metrics

Evaluation orchestration and metric computation are implemented in the standalone `eval/` module. See `specs/modules/spec_eval.md` for:

- `BaseEvaluator` / `BaseMetrics` ABCs
- `ClassifyEvaluator`, `DetectEvaluator`, `SegmentEvaluator`
- `ClassificationMetrics` (confusion matrix → Accuracy / Precision / Recall / F1)
- DataFlow-CV bridge contract (also see `specs/evaluate/spec_evaluate_bridge.md`)

## 6. Pipeline Factories

Pipeline factories use **direct construction with lazy backend imports** — no Registry, no decorators.
Each backend module is imported only when the user requests it, so only that backend's dependencies are needed.

**Design contract:**
- `_ensure_backend(name)`: triggers lazy import of the requested backend module
- `_build_backend(name, model_path, class_list, task_type, half, device)`: direct construction, returns `BaseBackend`
- Each pipeline factory (`create_detect_pipeline`, etc.) composes preprocessor + backend + postprocessor into an `InferencePipeline`

To add a new backend: create the backend file, add entries in `_ensure_backend` and `_build_backend`. No Registry, no decorators — each backend is independently extractable.

**Extension contract for adding a new backend:**
1. Create backend file in `modelflow/backends/`, inheriting from `BaseBackend` and implementing `__call__`
2. Add lazy-import + construction entries in each pipeline factory's `_ensure_backend()` / `_build_backend()`

**Extension contract for adding a new vision task:**
1. Create processor pair in `modelflow/processors/<task>/` (preprocessor + postprocessor)
2. Create pipeline factory in `modelflow/pipelines/<task>.py`
3. Add evaluator in `eval/evaluators/` (constructor injection, zero import dependency — see `spec_eval.md`)

## 7. Data Types

Task and backend types are identified by **string tags** (e.g., `"detect"`, `"onnxruntime"`, `"numpy"`). No enum classes are defined — strings are used directly in function signatures and configs. This avoids the maintenance burden of keeping enum members in sync with actual implementations.

```python
# modelflow/types.py
@dataclass
class ModelInfo:
    """Model input/output metadata."""
    name: str
    shape: List[int]
    dtype: np.dtype
    index: int = 0
```

## 8. Dependency Contract

### 8.1 Allowed Imports

The `modelflow/` package may import from:

| Source | Used By | Purpose |
|--------|---------|---------|
| `numpy` | All modules | Array operations |
| `onnxruntime` | `backends/onnx.py` | ONNX inference session |
| `tensorrt` | `backends/tensorrt.py` | TensorRT execution |
| `tritonclient` | `backends/triton.py` | Triton inference client |
| `torch` / `torchvision` | `processors/*/tch*.py` | PyTorch preprocessing (optional) |
| `PIL` / `cv2` | `processors/` | Image I/O |

### 8.2 Forbidden Imports

The `modelflow/` package must NOT import from:

| Forbidden Source | Reason |
|-----------------|--------|
| `export/*` | **Architecture constraint**: `modelflow/` ↔ `export/` zero cross-dependency |
| `data/*` | **Architecture constraint**: `modelflow/` is a pure inference engine, not a data loader |
| `eval/*` | **Architecture constraint**: `modelflow/` has zero knowledge of evaluation |
| `dataflow.*` | **Architecture constraint**: evaluation logic belongs in `eval/` only |
| `pycocotools` | **Architecture constraint**: evaluation logic belongs in `eval/` only |

### 8.3 Intra-Module Import Rules

```
pipelines/ → interfaces, types, processors/, backends/
processors/ → interfaces (BasePreprocessor, BasePostprocessor)
backends/ → interfaces (BaseBackend), types (ModelInfo) — never imports processors/
```

## 9. Error Handling Contract

### 9.1 Backend Errors

| Error Scenario | Behavior |
|---------------|----------|
| Model file not found | Raise `FileNotFoundError` with path |
| ONNX Runtime session creation failed | Raise `RuntimeError` with provider info |
| TensorRT engine deserialization failed | Raise `RuntimeError` |
| Triton server unreachable | Raise `ConnectionError` with server URL |
| Inference execution timeout | Raise `TimeoutError` |
| Output tensor count mismatch | Raise `ValueError` with expected vs actual |

### 9.2 Processor Errors

| Error Scenario | Behavior |
|---------------|----------|
| Empty input image (zero-size dimension) | Raise `ValueError` |
| Unsupported image channels (not 3) | Raise `ValueError` |
| Raw output shape mismatch with expected YOLO format | Raise `ValueError` with actual shape |
| Empty postprocessing result (no detections above threshold) | Return empty arrays — `boxes=(0,4)`, `scores=(0,)`, `class_ids=(0,)` |
| Unknown `model_version` string | Raise `ValueError` with valid options |

### 9.3 Evaluator Errors

Evaluator error handling is defined in the `eval/` module. See `specs/modules/spec_eval.md`.

## 10. See Also

- [`spec_architecture.md`](spec_architecture.md) — module architecture, Pipeline pattern
- [`spec_eval.md`](spec_eval.md) — evaluation module
