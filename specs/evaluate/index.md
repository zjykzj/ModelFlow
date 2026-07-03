# Evaluate Layer — Specification Index

> **Status:** Implemented
> **Version:** 1.0 | **Last Updated:** 2026-07-03

## What This Layer Covers

The Evaluate layer defines **WHAT** constitutes correct model evaluation in ModelFlow.
It documents the evaluation architecture, the bridge contract with DataFlow-CV,
and the prediction format requirements.

ModelFlow's evaluation is **split across three layers**:

- **samples/** (assembly layer) assembles pipeline + dataset and injects them into evaluators
- **eval/** (orchestration layer) runs inference, collects predictions, delegates to DataFlow-CV
- **DataFlow-CV** owns the metric computation (COCO → mAP/AP/AR/PRF1)

This layer defines the contract between `eval/` and DataFlow-CV.

## Layer Architecture

```
Evaluate Layer (WHAT)                Implementations
─────────────────────              ─────────────────────
spec_evaluate_bridge.md    ──▶    eval/evaluators/detect.py
                                 eval/evaluators/segment.py
                                 eval/evaluators/classify.py

  "What is the contract?"          "How does the code achieve it?"
```

## Relationship to Other Layers

```
specs/
├── evaluate/                       # WHAT — evaluation contracts (THIS LAYER)
│   ├── index.md                    #   You are here
│   └── spec_evaluate_bridge.md     #   ModelFlow ↔ DataFlow-CV bridge contract
│
├── modules/                        # HOW — module architecture
│   ├── spec_python.md              #   Evaluator base class, evaluation flow
│   └── ...
```

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`spec_evaluate_bridge.md`](spec_evaluate_bridge.md) | **Bridge contract** — ModelFlow ↔ DataFlow-CV integration: prediction format, API calls, graceful degradation, error handling |

## Reading Order

- **Adding a new evaluator task?** Read `spec_evaluate_bridge.md` first, then `specs/modules/spec_eval.md` for the evaluator base class and module structure.
- **Debugging mAP results?** Read `spec_evaluate_bridge.md` §2 (Prediction Format) to verify COCO format compliance.
- **DataFlow-CV API changed?** Update `spec_evaluate_bridge.md` §3 (Bridge API Contract) and verify the fallback path in §4.

## Evaluation Architecture at a Glance

```
                         samples/ (assembly layer)
                    assembles pipeline + dataset
                              │
            ┌─────────────────┼─────────────────┐
            │ constructor injection              │
            ▼                                    ▼
┌──────────────────────────────────────────────────────┐
│                     eval/ 模块                        │
│  ┌──────────────────────────────────────────────┐   │
│  │ Evaluator(pipeline, dataset, gt_json)        │   │
│  │   ├── ClassifyEvaluator → local metrics      │   │
│  │   ├── DetectEvaluator  → DataFlow-CV (bbox)  │   │
│  │   └── SegmentEvaluator → DataFlow-CV (segm)  │   │
│  │                                              │   │
│  │   run()                                      │   │
│  │     ├── _run_inference()  → COCO pred list   │   │
│  │     └── delegate to DataFlow-CV              │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
                         │
                         │ evaluate(gt_json, pred_list)
                         ▼
┌──────────────────────────────────────────────────────┐
│                     DataFlow-CV                       │
│  ┌────────────────────┐   ┌───────────────────────┐  │
│  │ COCO load + verify │──▶│ COCOeval (pycocotools) │  │
│  └────────────────────┘   │ → 12 standard metrics  │  │
│                           │ → per-class breakdown   │  │
│                           └───────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **samples/ assembles, eval/ orchestrates, DataFlow-CV computes.** The assembly layer creates pipeline + dataset and injects them into evaluators. `eval/` runs inference internally (via pipeline) and collects COCO predictions. DataFlow-CV receives GT+DT and computes metrics. This avoids duplicating metric logic.

2. **COCO JSON as the exchange format.** All prediction data passed between ModelFlow and DataFlow-CV is in COCO JSON format (plain annotation list). See `spec_evaluate_bridge.md` §2.

3. **Classification is self-contained.** `ClassifyEvaluator` does NOT delegate to DataFlow-CV. It uses `eval/` module's own `ClassificationMetrics` (confusion matrix based). Only detection and segmentation are bridged.

4. **Graceful degradation is mandatory.** If DataFlow-CV is not installed, evaluators must not crash. They return `{"num_predictions": N}` and log a warning.

5. **Segment evaluator uses SegmentationEvaluator.** `SegmentEvaluator` delegates to DataFlow-CV's `SegmentationEvaluator` (not `DetectionEvaluator`). Detection and segmentation use separate DataFlow-CV classes, each with the appropriate `iouType` (`'bbox'` vs `'segm'`).
