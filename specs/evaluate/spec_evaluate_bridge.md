# Evaluate Bridge Contract: ModelFlow ↔ DataFlow-CV

> **Version:** 1.0
> **Status:** Implemented
> **Layer:** Evaluate
> **Dependencies:** DataFlow-CV `specs/evaluate/spec_evaluate_metrics.md` (metric definitions), DataFlow-CV `specs/modules/spec_evaluate.md` (module API)

## 1. Document Scope

This document defines the **bridge contract** between ModelFlow's `eval/` module and DataFlow-CV (metric computation). It specifies:

- The COCO prediction format that `eval/` accepts as DT input
- The DataFlow-CV API that `eval/` evaluators delegate to
- The graceful degradation behavior when DataFlow-CV is unavailable
- Error handling and input validation rules

## 2. Prediction Format (ModelFlow → DataFlow-CV)

### 2.1 COCO Prediction List

ModelFlow evaluators produce a **plain COCO annotation list** (Format B in DataFlow-CV spec) — a JSON array of annotation dicts:

```json
[
  {
    "image_id": 0,
    "category_id": 1,
    "bbox": [x, y, width, height],
    "score": 0.85
  },
  ...
]
```

**No `images` or `categories` arrays at top level.** These are sourced from GT at load time via `COCO.loadRes()`.

### 2.2 Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_id` | int | **Yes** | 0-indexed image identifier. Matches `images[].id` in GT COCO JSON. |
| `category_id` | int | **Yes** | 1-indexed COCO category ID. ModelFlow internal class IDs (0-indexed) **must** be offset by +1. |
| `bbox` | [float×4] | **Yes** | COCO-format bounding box: `[x_tl, y_tl, width, height]` in absolute pixel coordinates. **Must be converted from `[x1, y1, x2, y2]` to `[x1, y1, x2-x1, y2-y1]`.** |
| `score` | float | **Yes** | Detection confidence in `[0.0, 1.0]`. Higher = more confident. **DataFlow-CV will reject annotations missing this field.** |

### 2.3 Optional Fields (Segmentation)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `segmentation` | object | No (detection) / Yes (seg) | Binary mask as list: `{"size": [h, w], "counts": mask.tolist()}` where `mask` is a binary `ndarray`. Presence triggers `iouType='segm'` in evaluation. |

### 2.4 Coordinate Conversion (Critical)

```
Internal (ModelFlow pipeline output):  [x1, y1, x2, y2] — top-left, bottom-right, absolute pixels
COCO (prediction format):              [x_tl, y_tl, width, height] — top-left, width, height, absolute pixels

Conversion:  x_tl = x1,  y_tl = y1,  width = x2 - x1,  height = y2 - y1
```

The conversion must happen inside `_run_inference()`, before the dict is appended to the predictions list.

### 2.5 Confidence Threshold

Detect evaluator: `conf_thres=0.001` (near-zero — let DataFlow-CV's PR curve sweep handle filtering).

Segment evaluator: same threshold via `pipeline(image, conf_thres=0.001, iou_thres=0.5)`.

**Why 0.001 and not 0?** A purely zero threshold includes boxes with zero confidence (invalid predictions). Near-zero captures all meaningful predictions while excluding degenerate outputs.

## 3. DataFlow-CV Bridge API

### 3.1 Import Path

```python
from dataflow.evaluate import DetectionEvaluator      # For detection (bbox IoU)
from dataflow.evaluate import SegmentationEvaluator   # For segmentation (mask IoU)
```

### 3.2 DetectionEvaluator

```python
class DetectionEvaluator(BaseEvaluator):
    """Object detection evaluation using bounding box IoU (iouType='bbox')."""

    def __init__(self, log_config=None):
        """
        Args:
            log_config: Optional[LogConfig]. If None, uses default
                LogConfig(name="evaluate"). Per-class metrics are computed
                only when log_config.verbose=True.
        """

    def evaluate(self, gt_source, dt_source) -> EvaluationResult:
        """
        Run full COCO evaluation (12 standard metrics).

        Args:
            gt_source: Union[str, Path, Dict, DatasetAnnotations]
            dt_source: Union[str, Path, List, Dict, DatasetAnnotations]

        Returns:
            EvaluationResult with .metrics (12 COCO metrics) and
            .per_class (per-category breakdown, None if not verbose).
        """
```

### 3.3 SegmentationEvaluator

```python
class SegmentationEvaluator(BaseEvaluator):
    """Instance segmentation evaluation using mask IoU (iouType='segm')."""

    def __init__(self, log_config=None):
        """Same constructor contract as DetectionEvaluator."""

    def evaluate(self, gt_source, dt_source) -> EvaluationResult:
        """Same evaluate contract. Uses iouType='segm' internally."""
```

### 3.4 ModelFlow eval/ Module Integration

ModelFlow evaluators construct DataFlow-CV evaluators with `LogConfig` (when available) and call `evaluate(gt_json, predictions)`. The returned `EvaluationResult` dataclass provides `metrics` (12 COCO standard metrics) and `per_class` (per-category breakdown when verbose).

The `dt_source` parameter accepts both a list (in-memory) and a string/Path (file):

| `dt_source` type | DataFlow-CV behavior |
|-----------------|---------------------|
| `List[dict]` | Directly passed to `coco_gt.loadRes(list)` |
| `str` / `Path` → file contains list | Passed to `coco_gt.loadRes(str(path))` |
| `str` / `Path` → file contains dict | Loaded via `pycocotools.COCO(path)` as full COCO dict |

ModelFlow's `DetectEvaluator.run()` uses `save_pred_json or predictions` — when a path is provided, DataFlow-CV loads the file; otherwise, the in-memory list is used directly.

## 4. Graceful Degradation Contract

### 4.1 Import Failure

**Mandatory behavior:**
- Catch `ImportError` specifically (not bare `except`)
- Log a warning message
- Return `{"num_predictions": len(predictions)}` so callers get a non-empty dict
- Do NOT crash or raise

### 4.2 Missing GT

**Mandatory behavior:**
- Return `{"num_predictions": N}` when GT is absent
- This is normal operation, not an error — some use cases only need prediction collection

## 5. Evaluator Contracts

### 5.1 DetectEvaluator

| Aspect | Contract |
|--------|----------|
| **Constructor** | `DetectEvaluator(pipeline, dataset, metrics=None, config=None, gt_json=None)` |
| **API** | `run(save_pred_json=None)` — runs inference internally, collects COCO predictions, delegates to DataFlow-CV |
| **DT format** | COCO annotation list: `[{"image_id", "category_id", "bbox": [x,y,w,h], "score"}, ...]` |
| **DataFlow-CV class** | `DetectionEvaluator` (iouType='bbox') |
| **Return** | `Dict[str, float]` with 12 COCO metrics, or `{"num_predictions": N}` on fallback |

### 5.2 SegmentEvaluator

| Aspect | Contract |
|--------|----------|
| **Constructor** | `SegmentEvaluator(pipeline, dataset, metrics=None, config=None, gt_json=None)` |
| **API** | `run(save_pred_json=None)` — runs inference internally, collects COCO predictions with masks, delegates to DataFlow-CV |
| **DT format** | COCO annotation list with `"segmentation"` field: `{"size": [h,w], "counts": mask.tolist()}` |
| **DataFlow-CV class** | `SegmentationEvaluator` (iouType='segm') |
| **Return** | Same as DetectEvaluator |

### 5.3 ClassificationMetrics (No Bridge)

| Aspect | Contract |
|--------|----------|
| **API** | In-memory: `update(pred_class, gt_class)` → `compute()` |
| **Metrics** | Self-contained — confusion matrix → Accuracy/P/R/F1 |
| **DataFlow-CV** | NOT used |

## 6. Architecture Constraint

```
ModelFlow (eval/)                  DataFlow-CV
───────────────────                ───────────
eval/evaluators/                   dataflow/evaluate/
├── detect.py ───────────────────▶ DetectionEvaluator
└── segment.py ──────────────────▶ SegmentationEvaluator

eval/ is a pure evaluation module — it does NOT import from
modelflow/ or data/. It accepts pipeline + dataset via
constructor injection and orchestrates inference internally.
```

**Hard constraints:**

1. **eval/ → DataFlow-CV: one-way dependency.** DataFlow-CV never imports from ModelFlow.
2. **Evaluator → DataFlow-CV: import guarded.** All `from dataflow.evaluate import ...` must be inside try/except ImportError blocks.
3. **Evaluator → DataFlow-CV: only public API.** Evaluators call `DetectionEvaluator.evaluate()` or `SegmentationEvaluator.evaluate()`.
4. **ClassificationMetrics must NOT import DataFlow-CV.** Classification metrics are self-contained.

## 7. Error Handling Contract

| Error Scenario | Behavior |
|---------------|----------|
| DataFlow-CV not installed | `ImportError` → log warning → return `{"num_predictions": N}` |
| `gt_json` is `None` or `""` | Log warning → return `{"num_predictions": N}` |
| `gt_json` file not found | DataFlow-CV raises → **propagate exception** (caller's responsibility to ensure file exists) |
| DT contains no predictions (`len(predictions) == 0`) | Pass to DataFlow-CV as empty list → `EvaluationResult(success=True, metrics=EvaluationMetrics(all=-1.0))` |
| DT missing `score` field | DataFlow-CV validation fails → `EvaluationResult(success=False, errors=[...])` |
| `category_id` in DT not in GT | DataFlow-CV logs warning → those DTs are excluded from matching |

## 8. See Also

- [`specs/modules/spec_eval.md`](../modules/spec_eval.md) — eval module architecture
- [DataFlow-CV](https://github.com/zjykzj/DataFlow-CV) — metric computation engine
