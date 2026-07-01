# Eval Module Specification

> **Version:** 0.4
> **Status:** Implemented
> **Dependencies:** `spec_architecture.md` (module relationships), `specs/evaluate/spec_evaluate_bridge.md` (DataFlow-CV bridge)

## 1. Overview

The `eval/` module provides **pure evaluation** — metric computation via pipeline + dataset orchestration.
It has **zero import dependency** on `modelflow/` or `data/` — dependencies are injected via constructor
(duck-typed interfaces, no hard `import modelflow` or `import data` statements).

```
eval/ → DataFlow-CV (optional, guarded import)
eval/ → utils/ (logger, ops)
eval/ → numpy, tqdm (standard packages)
eval/ → ZERO import dependency on modelflow/ or data/
```

### 1.1 Architecture Principle

Evaluators accept `pipeline` (InferencePipeline from modelflow) and `dataset` (from data) via constructor
injection. This is **dependency inversion**: `eval/` depends on the interface protocol, not the implementation.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   data/      │     │  modelflow/  │     │    eval/     │
│  (dataset)   │     │  (pipeline)  │     │ (evaluation) │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │     samples/ assembles & injects        │
       └────────────────────┼────────────────────┘
                            │
               samples/eval_detect.py
               samples/eval_classify.py
               samples/eval_segment.py
```

### 1.2 Directory Structure

```
eval/
├── __init__.py               # Public API exports
├── interfaces.py             # BaseEvaluator(pipeline, dataset, ...), BaseMetrics ABCs
├── evaluators/               # Evaluator implementations (direct construction)
│   ├── __init__.py
│   ├── base.py               # Re-export BaseEvaluator from interfaces
│   ├── classify.py           # ClassifyEvaluator(pipeline, dataset, metrics, config)
│   ├── detect.py             # DetectEvaluator(pipeline, dataset, metrics, config, gt_json)
│   └── segment.py            # SegmentEvaluator(pipeline, dataset, metrics, config, gt_json)
├── metrics/                  # Local metric implementations
│   ├── __init__.py
│   └── classification.py     # ClassificationMetrics(num_classes)
└── tests/
    ├── __init__.py
    ├── test_eval.py           # BaseEvaluator/BaseMetrics ABC tests
    └── test_metrics.py        # ClassificationMetrics tests
```

## 2. Core Interfaces

### 2.1 BaseEvaluator

```python
class BaseEvaluator(ABC):
    """评估器基类 — 编排 Pipeline + Dataset + Metrics"""

    def __init__(self, pipeline, dataset, metrics=None, config=None):
        self.pipeline = pipeline   # InferencePipeline (duck-typed, not imported)
        self.dataset = dataset     # Dataset (duck-typed, not imported)
        self.metrics = metrics     # BaseMetrics or None (detection delegates to DataFlow-CV)
        self.config = config or {}

    @abstractmethod
    def run(self) -> Dict[str, float]:
        """运行完整评估流程，返回 metrics dict"""
        ...
```

### 2.2 BaseMetrics

```python
class BaseMetrics(ABC):
    """指标计算基类（有状态累加器）"""

    @abstractmethod
    def update(self, prediction, ground_truth):
        """累加一个样本的指标"""
        ...

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """计算当前累加结果"""
        ...

    @abstractmethod
    def reset(self):
        """重置状态"""
        ...
```

## 3. Evaluators

### 3.1 ClassifyEvaluator

Uses local `ClassificationMetrics` (DataFlow-CV does not provide classification metrics).

```python
from modelflow.pipelines import create_classify_pipeline
from data import build_dataset
from eval import ClassifyEvaluator

pipeline = create_classify_pipeline(
    model_path="efficientnet_b0.onnx",
    class_list=get_class_names("imagenet"),
    backend="onnxruntime",
    input_size=224,
)
dataset = build_dataset("classify_dir", path="/data/imagenet/val")
evaluator = ClassifyEvaluator(pipeline, dataset)
results = evaluator.run()
# {"accuracy": 0.85, "precision_macro": 0.83, "recall_macro": 0.82, "f1_macro": 0.82}
```

**Constructor:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `pipeline` | InferencePipeline | yes | Classification pipeline (duck-typed) |
| `dataset` | Dataset | yes | Dataset with `class_list` or `num_classes` attribute |
| `metrics` | ClassificationMetrics | no | Auto-created from `dataset.num_classes` if not provided |
| `config` | dict | no | Optional evaluation config |

**`run()`** iterates the dataset, calls `pipeline(image)` for each sample, feeds results to `ClassificationMetrics.update()`, returns `metrics.compute()`.

### 3.2 DetectEvaluator

Orchestrates pipeline inference internally, collects COCO-format predictions, delegates mAP computation to DataFlow-CV.

```python
from eval import DetectEvaluator

evaluator = DetectEvaluator(pipeline, dataset, gt_json="annotations.json")
results = evaluator.run()
# {"mAP": 0.503, "AP50": 0.698, ...}

# Optionally save predictions
results = evaluator.run(save_pred_json="preds.json")
```

**Constructor:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `pipeline` | InferencePipeline | yes | Detection pipeline with `preprocessor` and `backend` |
| `dataset` | Dataset | yes | Dataset returning `(image, gt)` tuples |
| `metrics` | — | no | Not used — detection delegates to DataFlow-CV |
| `config` | dict | no | Optional evaluation config |
| `gt_json` | str | no | Path to COCO GT JSON (or auto-detected via `dataset.get_gt_json()`) |

**Internal flow:**

```
run(save_pred_json=None)
    │
    ├── 1. _run_inference()
    │       Iterate dataset: image, gt = dataset[idx]
    │       preprocess → backend infer → decode raw outputs
    │       Convert to COCO prediction format:
    │         {"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}
    │       Return List[dict]
    │
    ├── 2. Save predictions (if save_pred_json)
    │
    ├── 3. Delegate to DataFlow-CV
    │       from dataflow.evaluate import DetectionEvaluator
    │       df_eval = DetectionEvaluator(log_config=LogConfig(name="eval", verbose=True))
    │       result = df_eval.evaluate(gt_json, predictions)
    │       Return _to_metrics(result)
    │
    └── 4. Fallback (no gt_json or DataFlow-CV unavailable)
            Return {"num_predictions": len(predictions)}
```

**Returns:** `Dict[str, float]` — 12 COCO standard metrics:

```python
{
    "mAP": 0.503, "AP50": 0.698, "AP75": 0.547,
    "AP_small": 0.312, "AP_medium": 0.501, "AP_large": 0.612,
    "AR_max_1": 0.321, "AR_max_10": 0.521, "AR_max_100": 0.538,
    "AR_small": 0.398, "AR_medium": 0.567, "AR_large": 0.689,
}
```

When DataFlow-CV is unavailable: `{"num_predictions": N}`.

### 3.3 SegmentEvaluator

Orchestrates segmentation pipeline inference, collects COCO-format predictions with masks, delegates to DataFlow-CV's `SegmentationEvaluator` (`iouType='segm'`).

```python
from eval import SegmentEvaluator

evaluator = SegmentEvaluator(pipeline, dataset, gt_json="annotations.json")
results = evaluator.run()
```

**Constructor:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `pipeline` | InferencePipeline | yes | Segmentation pipeline with full postprocessor |
| `dataset` | Dataset | yes | Dataset returning `(image, gt)` tuples |
| `gt_json` | str | no | Path to COCO GT JSON (or auto-detected) |

**Internal flow:** Same pattern as `DetectEvaluator`, but:
1. Uses `pipeline(image, conf_thres=0.001, iou_thres=0.5)` for full postprocessing
2. Adds `"segmentation"` field: `{"size": [h, w], "counts": mask.tolist()}` (binary mask as list)
3. Delegates to `dataflow.evaluate.SegmentationEvaluator` (not DetectionEvaluator)

### 3.4 ClassificationMetrics

Local implementation — DataFlow-CV does not provide classification metrics.

```python
from eval.metrics import ClassificationMetrics

m = ClassificationMetrics(num_classes=1000)
for pred, gt in inference_loop:
    m.update(pred, gt)
result = m.compute()
# {"accuracy": 0.85, "precision_macro": 0.83, "recall_macro": 0.82, "f1_macro": 0.82}
m.reset()
```

| Method | Description |
|--------|-------------|
| `update(prediction, ground_truth)` | Accumulate one sample into confusion matrix. `prediction["class_ids"]` is list of predicted class IDs. `ground_truth["class_id"]` is the ground truth class ID. |
| `compute()` | Compute Accuracy / Precision(macro) / Recall(macro) / F1(macro) from confusion matrix |
| `reset()` | Zero out the confusion matrix |

## 4. DataFlow-CV Bridge

Detection and segmentation evaluators delegate mAP computation to DataFlow-CV via a guarded import pattern:

```python
try:
    from dataflow.evaluate import DetectionEvaluator  # or SegmentationEvaluator
    from dataflow.util.logging import LogConfig

    df_eval = DetectionEvaluator(
        log_config=LogConfig(name="eval", verbose=True)
    )
    result = df_eval.evaluate(gt_json, predictions)
    return _to_metrics(result)
except ImportError:
    logger.warning("DataFlow-CV not installed. ...")
    return {"num_predictions": len(predictions)}
```

Key design decisions:
- **Classification** uses local `ClassificationMetrics` — DataFlow-CV has no classification module
- **Detection** delegates to `dataflow.evaluate.DetectionEvaluator`
- **Segmentation** delegates to `dataflow.evaluate.SegmentationEvaluator` (separate class, `iouType='segm'`)
- **Guarded import** — `from dataflow.evaluate import ...` is always inside try/except
- **Fallback** — returns `{"num_predictions": N}` when DataFlow-CV is not installed

See `specs/evaluate/spec_evaluate_bridge.md` for the full DataFlow-CV interface contract.

## 5. Extension

Evaluators use direct construction — no Registry, no decorators:

```python
# Adding a new evaluator:
# 1. Create eval/evaluators/pose.py
class PoseEvaluator(BaseEvaluator):
    def __init__(self, pipeline, dataset, metrics=None, config=None,
                 gt_json=None):
        super().__init__(pipeline, dataset, metrics, config)
        self.gt_json = gt_json

    def run(self) -> Dict[str, float]:
        # ... inference + metrics logic
        ...

# 2. Register in eval/evaluators/__init__.py
from .pose import PoseEvaluator

# 3. Use directly
evaluator = PoseEvaluator(pipeline, dataset)
results = evaluator.run()
```

## 6. Dependency Contract

### 6.1 Allowed Imports

| Source | Used By | Purpose |
|--------|---------|---------|
| `numpy` | metrics | Array operations |
| `tqdm` | evaluators | Progress bars |
| `utils.logger` | evaluators | Logging |
| `utils.ops` | evaluators/detect.py | `xywh2xyxy` coordinate conversion |
| `dataflow.evaluate` | evaluators/detect.py, evaluators/segment.py | mAP computation (**guarded**) |
| `dataflow.util.logging` | evaluators/ | LogConfig (**guarded**) |

### 6.2 Forbidden Imports

| Forbidden Source | Reason |
|-----------------|--------|
| `modelflow/*` | `eval/` uses duck-typed pipeline interface — zero import dependency |
| `data/*` | `eval/` uses duck-typed dataset interface — zero import dependency |
| `export/*` | No cross-dependency with export |
| `dataflow.*` (unguarded) | Must be inside `try/except ImportError` |

### 6.3 Constructor Injection Contract

Evaluators accept `pipeline` and `dataset` as constructor parameters. These are typed as duck-typed interfaces (not imported from `modelflow` or `data`):

- **pipeline**: Must have `preprocessor`, `backend`, and be callable as `pipeline(image)` → `dict`. The `DetectEvaluator` accesses `pipeline.preprocessor` and `pipeline.backend` directly for performance.
- **dataset**: Must support `len(dataset)` and `dataset[idx]` → `(image, gt_dict)`. May optionally provide `get_gt_json()` for auto-detecting GT annotation path.

## 7. See Also

- [`specs/SDD_GUIDE.md`](../SDD_GUIDE.md) — full change history
- [`specs/evaluate/spec_evaluate_bridge.md`](../evaluate/spec_evaluate_bridge.md) — DataFlow-CV bridge contract
- [`spec_architecture.md`](spec_architecture.md) — module architecture, dependency contract
