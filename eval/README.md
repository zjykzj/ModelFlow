# Eval Module

> Evaluation orchestration and metrics computation — depends on `modelflow/` (Pipeline) and `data/` (Dataset).

<br>

## 1. Overview

The `eval/` module provides evaluation infrastructure for ModelFlow. It orchestrates the inference loop, collects predictions, and computes metrics — either locally (classification) or by delegating to [DataFlow-CV](https://github.com/zjykzj/DataFlow-CV) (detection/segmentation).

**Module dependency direction:** `eval/` → `modelflow/` + `data/`. `modelflow/` has zero knowledge of `eval/`.

| Component | Role |
|-----------|------|
| `eval/interfaces.py` | `BaseEvaluator` / `BaseMetrics` ABCs |
| `eval/evaluators/` | `ClassifyEvaluator`, `DetectEvaluator`, `SegmentEvaluator` |
| `eval/metrics/` | `ClassificationMetrics` (confusion matrix → Accuracy/Precision/Recall/F1) |

<br>

## 2. Quick Start

### 2.1 Classification Evaluation

```python
from eval import ClassifyEvaluator
from data import build_dataset, get_class_names

# Build pipeline (modelflow)
from modelflow.pipelines import create_classify_pipeline
class_list = get_class_names("imagenet")
pipeline = create_classify_pipeline(
    model_path="efficientnet_b0.onnx",
    class_list=class_list,
    backend="onnxruntime",
    input_size=224,
)

# Build dataset (data)
dataset = build_dataset("imagenet", path="/data/imagenet", val="val")

# Evaluate
evaluator = ClassifyEvaluator(pipeline, dataset)
metrics = evaluator.run()
# {"accuracy": 0.85, "precision_macro": 0.83, "recall_macro": 0.82, "f1_macro": 0.82}
```

### 2.2 Detection Evaluation

```python
from eval import DetectEvaluator
from data import build_dataset, get_class_names

from modelflow.pipelines import create_detect_pipeline
class_list = get_class_names("coco")
pipeline = create_detect_pipeline(
    model_path="yolov8s.onnx",
    class_list=class_list,
    backend="onnxruntime",
)

dataset = build_dataset(
    "coco", path="/data/coco", val="val2017",
    anno="annotations/instances_val2017.json",
)

evaluator = DetectEvaluator(pipeline, dataset, gt_json=dataset.get_gt_json())
metrics = evaluator.run()
# If DataFlow-CV available: {"mAP": 0.50, "AP50": 0.70, ...}
# Fallback: {"num_predictions": N}
```

### 2.3 Instance Segmentation Evaluation

```python
from eval import SegmentEvaluator
from data import build_dataset, get_class_names

from modelflow.pipelines import create_segment_pipeline
class_list = get_class_names("coco-seg")
pipeline = create_segment_pipeline(
    model_path="yolov8s-seg.onnx",
    class_list=class_list,
)

dataset = build_dataset(
    "coco-seg", path="/data/coco", val="val2017",
    anno="annotations/instances_val2017.json",
)

evaluator = SegmentEvaluator(pipeline, dataset, gt_json=dataset.get_gt_json())
metrics = evaluator.run()
```

### 2.4 Save Predictions

```python
# Save COCO-format predictions to file
metrics = evaluator.run(save_pred_json="predictions.json")
```

<br>

## 3. Evaluator Reference

### 3.1 ClassifyEvaluator

Self-contained — uses local `ClassificationMetrics`. No DataFlow-CV dependency.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | `InferencePipeline` | required | Classify pipeline |
| `dataset` | `BaseDataset` | required | Classification dataset |
| `metrics` | `ClassificationMetrics` | auto-created | Metrics accumulator |
| `config` | `dict` | `{}` | Optional config |

**Inference method:** `pipeline(image)` — full pipeline end-to-end.

### 3.2 DetectEvaluator

Delegates mAP to DataFlow-CV. Falls back to `{"num_predictions": N}` when DataFlow-CV is unavailable.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | `InferencePipeline` | required | Detect pipeline |
| `dataset` | `BaseDataset` | required | COCO-format image dataset |
| `gt_json` | `Optional[str]` | from dataset | COCO GT annotation JSON path |
| `config` | `dict` | `{}` | Optional config |

**Inference method:** Manual — `pipeline.preprocessor(image)` → `pipeline.backend(tensor)` → manual decode.
This bypasses the postprocessor to collect raw COCO-format predictions.

### 3.3 SegmentEvaluator

Same API as `DetectEvaluator`. Delegates mAP to DataFlow-CV. Includes RLE mask data in COCO predictions.

**Inference method:** `pipeline(image, conf_thres=0.001, iou_thres=0.5)` — full pipeline with mask decoding.

<br>

## 4. Metrics

### 4.1 ClassificationMetrics

Confusion matrix-based metrics accumulator. No DataFlow-CV dependency.

```python
from eval.metrics import ClassificationMetrics

m = ClassificationMetrics(num_classes=10)
m.update({"class_ids": [0]}, {"class_id": 0})
result = m.compute()
# {"accuracy": 1.0, "precision_macro": 1.0, "recall_macro": 1.0, "f1_macro": 1.0}
m.reset()
```

| Method | Description |
|--------|-------------|
| `update(prediction, ground_truth)` | Accumulate one sample into confusion matrix |
| `compute()` | Compute Accuracy / Precision(macro) / Recall(macro) / F1(macro) |
| `reset()` | Zero out the confusion matrix |

## 5. DataFlow-CV Bridge

Detection and segmentation evaluators delegate mAP computation to DataFlow-CV v1.5.0+ via a **guarded import**:

```python
try:
    from dataflow.evaluate import DetectionEvaluator
    from dataflow.util.logging import LogConfig

    df_eval = DetectionEvaluator(
        log_config=LogConfig(name="eval", verbose=True)
    )
    result = df_eval.evaluate(gt_json, predictions)

    # result is an EvaluationResult dataclass
    if result.success:
        m = result.metrics
        print(f"mAP={m.ap:.3f}, AP50={m.ap50:.3f}")
except ImportError:
    logger.warning("DataFlow-CV not installed.")
    return {"num_predictions": len(predictions)}
```

If DataFlow-CV is not installed, evaluation **never crashes** — it returns `{"num_predictions": N}` and logs a warning.

See [`specs/evaluate/spec_evaluate_bridge.md`](../specs/evaluate/spec_evaluate_bridge.md) for the full bridge contract.

<br>

## 6. Package Structure

```
eval/
├── __init__.py               # Public API exports
├── interfaces.py             # BaseEvaluator, BaseMetrics ABCs
├── evaluators/               # Evaluation orchestrators
│   ├── base.py               # Re-export BaseEvaluator
│   ├── classify.py           # ClassifyEvaluator (local metrics)
│   ├── detect.py             # DetectEvaluator (DataFlow-CV bridge)
│   └── segment.py            # SegmentEvaluator (DataFlow-CV bridge)
├── metrics/                  # Metric implementations
│   └── classification.py     # ClassificationMetrics (confusion matrix)
└── tests/
    ├── test_eval.py
    └── test_metrics.py
```

