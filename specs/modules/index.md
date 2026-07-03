# Modules Layer — Specification Index

> **Version:** 1.0 | **Last Updated:** 2026-07-03
> **Status:** Implemented — these documents define the implementation architecture of ModelFlow.

## What This Layer Covers

The Modules layer defines **HOW** ModelFlow is built. It describes the implementation contracts for:

- The top-level modules: `utils/` (shared utilities), `modelflow/` (inference engine), `data/` (datasets), `eval/` (evaluation), `export/` (model export), `vlms/` (CLIP/OpenCLIP)
- The Pipeline pattern that unifies all inference across backends
- Each module's internal structure, interfaces, extension points, and design rationale

## Architecture Diagram

```
                  ┌──────────────────────────┐
                  │        export /           │
                  │  onnx → tensorrt / triton │
                  └──────────────────────────┘
                        │           │
                        │ *.onnx    │ *.engine / triton config
                        ▼           ▼
 ┌──────────────────────────────────────────┐
 │              modelflow /                  │
 │  Pipeline(processor + backend + postproc) │
 │  ┌────────┐  ┌────────┐  ┌────────────┐  │
 │  │onnx    │  │trt     │  │triton      │  │
 │  └────────┘  └────────┘  └────────────┘  │
 │  Pure inference engine                   │
 └──────────────────┬───────────────────────┘
                    │ constructor injection
                    ▼
 ┌──────────────────────────────────────────┐
 │              eval /                       │
 │  Evaluator(pipeline, dataset) → run()    │
 │  └──▶ DataFlow-CV (guarded import)       │
 │  Classify / Detect / Segment              │
 └──────────────────────────────────────────┘
       ▲                        ▲
       │ constructor injection  │
       │                        │
 ┌─────┴───────┐    ┌───────────┴──────┐
 │   data /     │    │    utils /        │
 │ Dataset+YAML │    │ logger, ops       │
 └─────────────┘    └──────────────────┘

 ┌──────────────────────────────────────────┐
 │              vlms /                       │
 │  CLIP / OpenCLIP processors & samples    │
 │  (independent of modelflow/)             │
 └──────────────────────────────────────────┘
```

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`spec_architecture.md`](spec_architecture.md) | Module inventory, inter-module relationships, Pipeline pattern, data flow, implementation roadmap |
| 2 | [`spec_python.md`](spec_python.md) | `modelflow/` package: pure inference engine — ABCs (Backend/Processor/Pipeline), backends (ONNX/TensorRT/Triton), processors (4 tasks), Pipeline factories (lazy-import, direct construction) |
| 3 | [`spec_eval.md`](spec_eval.md) | `eval/` module: evaluation orchestration — BaseEvaluator/BaseMetrics ABCs, evaluators (Classify/Detect/Segment), ClassificationMetrics, DataFlow-CV bridge (constructor injection, direct construction) |
| 4 | [`spec_export.md`](spec_export.md) | `export/` module: pt → onnx → tensorrt/triton export pipeline stages, I/O specifications, precision validation standards |

> **Export knowledge layer supplement:** `spec_export.md` defines only the architectural blueprint. For format principles, conversion specifications, model differences, and design rationale, see the [`specs/export/`](../export/index.md) series ([`onnx_export.md`](../export/onnx_export.md), [`tensorrt_conversion.md`](../export/tensorrt_conversion.md), [`triton_deployment.md`](../export/triton_deployment.md)).

## Dependencies Between Modules

```
spec_architecture.md  (inter-module relationships, Pipeline pattern definition)
    │
    ├──▶ spec_python.md   (implements Pipeline contract — pure inference engine)
    ├──▶ spec_eval.md     (evaluation orchestration — accepts pipeline+dataset via constructor injection,
    │                      zero import dependency on modelflow/ or data/)
    ├──▶ spec_export.md   (export module architectural blueprint)
    │       │
    │       └──▶ specs/export/ series (principled knowledge layer extension)
```

## Reading Order

| Reader | Recommended Order |
|--------|-------------------|
| **New developer / AI Agent** | `CLAUDE.md` → this document |
| **Overall architecture understanding** | `spec_architecture.md` → `spec_python.md` |
| **Python inference development** | `spec_python.md` → `spec_architecture.md` |
| **Evaluation development** | `spec_eval.md` → [`specs/evaluate/spec_evaluate_bridge.md`](../evaluate/spec_evaluate_bridge.md) → `spec_architecture.md` |
| **Model export (architecture overview)** | `spec_export.md` |
| **Model export (principles understanding)** | `spec_export.md` → [`specs/export/`](../export/index.md) series |
