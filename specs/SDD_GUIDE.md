# SDD Development Guide

> **This document defines the ModelFlow project-specific SDD development guide.**
>
> Target audience: AI Agents working on this project (such as Claude Code). Also applicable to human developers.
>
> **Universal methodology: see [`SDD_METHODOLOGY.md`](SDD_METHODOLOGY.md).** This document focuses on ModelFlow-specific content and is the project engineering supplement to the universal methodology.

## Quick Start

**If you are an AI Agent starting work on ModelFlow:**

1. Read [`SDD_METHODOLOGY.md`](SDD_METHODOLOGY.md) first — understand the three-layer SDD system and the development workflow.
2. Then read this document (SDD_GUIDE.md) — learn ModelFlow's architecture, hard constraints, spec mapping, and common scenarios.
3. Use the **Specs Navigation Map** in §3 to find the right spec for your task.

---

## 1. Project Architecture

### Module System

ModelFlow's module dependency relationships (detailed in [`modules/index.md`](modules/index.md)):

```
                    ┌─────────────────┐
                    │     samples/     │
                    │ (infer / eval)   │
                    └──────┬──────────┘
                           │  calls pipeline factories & evaluators
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         modelflow/                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                     │
│  │pipelines │──>│processors│   │ backends │                     │
│  │(factory) │   │(pre/post)│   │(infer)    │                     │
│  └──────────┘   └──────────┘   └──────────┘                     │
│       │              │              │                            │
│       │              │   ZERO      │                            │
│       │              │   IMPORT    │                            │
│       │              │   DEPENDENCY│                            │
│       │              │  (one-way)  │                            │
│       │              │              │                            │
│       └──────┬───────┘              │                            │
│              │                      │                            │
│              ▼                      ▼                            │
│  interfaces.py + types.py + config.py                              │
│  (flattened at modelflow/ root — no core/ subpackage)            │
│                                                                  │
│         Pure inference engine — no eval, no metrics, no data     │
└─────────────────────────────────────────────────────────────────┘
                           │ Pipeline API
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         eval/                                     │
│  Evaluator(pipeline, dataset) → run()                            │
│  └──▶ DataFlow-CV (guarded import)                               │
│  ZERO import dependency on modelflow/ or data/                   │
│  (dependencies injected via constructor)                         │
└─────────────────────────────────────────────────────────────────┘
                           ▲ constructor injection
                           │
┌─────────────────┐
│     data/        │
│  Dataset + YAML  │
└─────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       export/ (ZERO dependency on modelflow/)    │
│  _base/_validation/_utils → onnx/ → tensorrt/ → triton/         │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Hard Constraints (Recite Before Writing Code)

| # | Constraint | Violation Consequence |
|---|------------|----------------------|
| 1 | **`modelflow/` ↔ `export/`: zero cross-dependency** | Module coupling, unable to develop independently |
| 2 | **Backend → Preprocessor/Postprocessor: zero reference** | Backend should not be aware of image preprocessing |
| 3 | **Backend only receives `np.ndarray`, only returns `List[np.ndarray]`** | Breaks Pipeline abstraction |
| 4 | **Processor → Backend: no direct calls** | Pipeline is the sole orchestrator |
| 5 | **`modelflow/` is a pure inference engine** — zero dependency on data/、eval/、DataFlow-CV | Inference code entangled with evaluation/data loading |
| 6 | **`eval/` is a pure evaluation module** — zero import dependency on modelflow/、data/ (dependencies injected via constructor) | Evaluation code entangled with inference/data loading |
| 7 | **`eval/` → DataFlow-CV: import guarded** by try/except ImportError | Crash when DataFlow-CV not installed |
| 8 | **`export/` → `modelflow/`: zero imports** | Export uses self-contained `export/_utils.py` preprocessing |

### Task Coverage

| Task | Python (`modelflow/`) | Export |
|------|-----------------------|--------|
| Classification | ✅ | ✅ |
| Detection | ✅ | ✅ |
| Instance Segmentation | ✅ | ✅ |
| Semantic Segmentation | ✅ | ✅ |
| Multi-modal (CLIP) | ✅ | ✅ |

---

## 2. Development Workflow (Project-Specific Supplement)

> The universal four-step workflow (scope of impact → spec → dev context → plan) is defined in [`SDD_METHODOLOGY.md`](SDD_METHODOLOGY.md) §2.

### 2.1 Step 1: Determine Scope of Impact (Project Template)

1. Which major module does the change affect? (`modelflow/` / `export/` / `eval/` / `data/` / `utils/` / `vlms/` / `samples/`)
2. Which Pipeline stage does it involve? (Preprocessor / Backend / Postprocessor)
3. Which task type does it involve? (Classify / Detect / Segment / SemanticSeg / Multimodal)

### 2.2 Step 2: Read Specs (Project Mapping Table)

Find the corresponding spec by change type:

| Change Type | Required Spec Reading |
|-------------|----------------------|
| Add/modify inference backend | `specs/modules/spec_python.md` (Section 4: Backend) |
| Add/modify preprocessing/postprocessing | `specs/modules/spec_python.md` (Section 5: Processor) |
| Add/modify Pipeline | `specs/modules/spec_python.md` (Section 3: InferencePipeline) |
| Add/modify evaluator | `specs/modules/spec_eval.md` (constructor injection + run() API) → `specs/evaluate/spec_evaluate_bridge.md` (DataFlow-CV bridge) |
| Add/modify dataset | `data/README.md` (data module docs) → `specs/modules/spec_architecture.md` |
| Add ONNX export | `specs/modules/spec_export.md` → `specs/export/onnx_export.md` |
| Add TensorRT export | `specs/modules/spec_export.md` → `specs/export/tensorrt_conversion.md` |
| Add Triton deployment | `specs/modules/spec_export.md` → `specs/export/triton_deployment.md` |
| Cross-module changes | `specs/modules/spec_architecture.md` (architecture constraint diagram + module dependency rules) |
| Unclear scope of impact | `specs/modules/spec_architecture.md` (global overview) |

### 2.3 Project-Specific Implementation Details

**Backend Contract (most error-prone area)**

```
Input:  np.ndarray — NCHW float32, already preprocessed by Preprocessor
Output: List[np.ndarray] — list of raw model outputs

❌ Backend must not perform any image processing internally (resize/normalize/letterbox)
❌ Backend must not perform any postprocessing internally (NMS/softmax/decoding)
✅ Backend does only one job: run inference and return raw outputs
```

**Preprocessor Data Flow**

```
Input: np.ndarray — HWC uint8 BGR (OpenCV standard loading format)

classify:  BGR→RGB → Resize(256) → CenterCrop(224) → Normalize → HWC→CHW → BatchDim
detect:    LetterBox → BGR→RGB → /255 → HWC→CHW
segment:   (same as detect)
semantic_seg: BGR→RGB → Resize(optional) → /255 → Normalize → HWC→CHW

```

**Postprocessor Data Flow**

```
Input: List[np.ndarray] — list of raw model outputs

classify:  softmax → top-k → {class_ids, scores, class_names?}
detect:    transpose → confidence threshold → xywh2xyxy → NMS → scale_boxes → {boxes, scores, class_ids}
segment:   confidence threshold → NMS → process_mask(proto + coeffs) → crop_mask → {boxes, scores, class_ids, masks}
semantic_seg: argmax → colormap(optional) → {class_map, colormap?}

```

**Pipeline Calling Convention**

```python
pipeline = InferencePipeline(preprocessor, backend, postprocessor)

# End-to-end inference (recommended)
result = pipeline(image, conf_thres=0.25, iou_thres=0.45)

# Skip preprocessing/postprocessing (for evaluation)
raw_outputs = pipeline.infer(tensor)
```

**Evaluator Flow (in `eval/` module — constructor injection, zero import dependency)**

```
ClassifyEvaluator(pipeline, dataset, metrics, config).run()
    → pipeline(image) per sample → ClassificationMetrics.update() → compute()

DetectEvaluator(pipeline, dataset, metrics, config, gt_json).run()
    → manual inference (preprocessor + backend) → COCO pred list → DataFlow-CV DetectionEvaluator

SegmentEvaluator(pipeline, dataset, metrics, config, gt_json).run()
    → pipeline(image) per sample → COCO pred list with masks → DataFlow-CV SegmentationEvaluator
```

samples/ assembles pipeline + dataset and injects them into the evaluator.

**Export Depth Levels**

| Level | Output | Runtime |
|-------|--------|---------|
| **L1** | `.onnx` | ONNX Runtime (CPU/GPU) |
| **L2** | `.onnx` + `.engine` | TensorRT GPU (FP16/INT8) |
| **L3** | `.onnx` + `.engine` + Triton config | Triton Inference Server |

```
Export data flow:
PT Model → BaseExporter.export_onnx() → .onnx → TensorRT Builder → .engine → Triton config → model repo
              ↓
        validation.check_onnx() + validation.compare_torch_onnx()
```

### 2.4 Before Submitting

```bash
# 1. Run tests (must pass)
pytest modelflow/tests/ eval/tests/ data/tests/ -v

# 2. If export-related code was changed, run export tests
pytest export/tests/ -v
```

**Documentation sync check (must do for every change):**

| Priority | Document | Check Condition | Action |
|----------|----------|----------------|--------|
| **P0** | `specs/` | Behavior changes (interface, contract, data flow) | **Must** sync update |
| **P1** | `CLAUDE.md` | New architecture details, new gotchas, new hard constraints, new key implementations | Update Known Gotchas or Critical Details |
| **P1** | `README.md` | API changes, new feature entry points, installation step changes | Sync update user documentation |
| **P2** | `samples/` | User API changes, new task types, calling convention changes | Update sample code (`samples/infer.py`, etc.) |

**Git commit format:**

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body if needed>

Co-Authored-By: DeepSeek-V4.0 <noreply@deepseek.com>
EOF
)"
```

Types: `feat` / `fix` / `docs` / `refactor` / `test` / `style` / `perf` / `chore`

---

## 3. Specs Navigation Map

### 3.1 Where Do I Find What?

```
"What is the Pipeline calling flow?"
  → specs/modules/spec_python.md (Section 3: InferencePipeline)

"What format does the Preprocessor output?"
  → specs/modules/spec_python.md (Section 5.2: Processor Input/Output Specification)

"What is the Backend __call__ contract?"
  → specs/modules/spec_python.md (Section 4.2: Backend Interface)

"How is NMS implemented?"
  → specs/modules/spec_python.md (Section 5.3: Processors → DetectPostprocessor)

"How is mask decoding done?"
  → specs/modules/spec_python.md (Section 5.3: Processors → SegmentPostprocessor)

"How do I add a new backend?"
  → specs/modules/spec_python.md (Section 4.3: Registering a New Backend)

"What metrics are available in evaluation results?"
  → specs/evaluate/spec_evaluate_bridge.md (Section 3: Bridge API)
  → specs/modules/spec_eval.md (Evaluator constructors + Metrics)

"How does ModelFlow integrate with DataFlow-CV for mAP?"
  → specs/evaluate/index.md (architecture overview)
  → specs/evaluate/spec_evaluate_bridge.md (full contract)

"What format should COCO predictions be in?"
  → specs/evaluate/spec_evaluate_bridge.md (Section 2: Prediction Format)

"What are the steps for ONNX export?"
  → specs/modules/spec_export.md (Section 3: Export Pipeline)

"How do I choose between FP16 and INT8 for TensorRT?"
  → specs/modules/spec_export.md (Section 4: Conversion Strategy)
  → specs/export/tensorrt_conversion.md (quantization decision tree)

"What is the directory structure of a Triton model repository?"
  → specs/export/triton_deployment.md (Section 2: Model Repository Structure)

"What are the dependency relationships between modules?"
  → specs/modules/spec_architecture.md (Architecture Constraint diagram)

```

### 3.2 Reading Order by Role

| Reader | Recommended Order |
|--------|-------------------|
| **Developer / AI Agent** | Start here → select module specs based on scope of impact |
| **Architecture Understanding** | `modules/spec_architecture.md` → `modules/spec_python.md` → `modules/spec_eval.md` |
| **Python Inference Development** | `modules/spec_python.md` → `modules/spec_architecture.md` |
| **Evaluation Development** | `modules/spec_eval.md` → `evaluate/spec_evaluate_bridge.md` |
| **Model Export (Architecture)** | `modules/spec_export.md` |
| **Model Export (Deep Dive)** | `modules/spec_export.md` → `export/onnx_export.md` → `export/tensorrt_conversion.md` → `export/triton_deployment.md` |

### 3.3 File Inventory

```
specs/
├── SDD_METHODOLOGY.md             # Universal SDD methodology (project-agnostic)
├── SDD_GUIDE.md                   # This document — ModelFlow-specific development guide
│
├── modules/                       # HOW — internal module architecture
│   ├── index.md                   # Modules layer overview
│   ├── spec_architecture.md       # Architecture overview: all modules + Pipeline pattern
│   ├── spec_python.md             # modelflow/ package: pure inference engine
│   ├── spec_eval.md               # eval/ package: evaluation orchestration & metrics
│   ├── spec_export.md             # Export module: pipeline, depth levels, precision verification
│
├── export/                        # WHAT — export format and conversion knowledge
│   ├── index.md                   # Export knowledge layer overview
│   ├── onnx_export.md             # ONNX export principles
│   ├── tensorrt_conversion.md     # TensorRT conversion and quantization
│   └── triton_deployment.md       # Triton deployment configuration
│
└── evaluate/                      # WHAT — evaluation bridge contract
    ├── index.md                   # Evaluate layer overview
    └── spec_evaluate_bridge.md    # ModelFlow ↔ DataFlow-CV bridge contract
```

---

## 4. Common Development Scenarios

### Scenario: Add a New Inference Backend

1. Confirm the Backend Interface contract in `specs/modules/spec_python.md`
2. Create `your_backend.py` in `modelflow/backends/`, inheriting from `BaseBackend`
3. Add a lazy-import entry in each pipeline factory's `_ensure_backend()` helper
4. Add a construction entry in each pipeline factory's `_build_backend()` helper
5. Add tests in `modelflow/tests/test_backends.py`
6. Add pipeline tests in `modelflow/tests/test_pipelines.py`

### Scenario: Add a New Task Type (e.g., pose)

1. Create a `pose/` sub-package in `modelflow/processors/` (preprocess.py + postprocess.py)
2. Create a `pose.py` factory function in `modelflow/pipelines/`
3. Add a corresponding evaluator in `eval/evaluators/` (pure evaluation — no Pipeline dependency, see `specs/modules/spec_eval.md`)
4. Add a task branch in `samples/infer.py` and `samples/eval_*.py`
5. Confirm export pipeline support in `export/`
6. Write tests

### Scenario: Add a New Dataset

1. Create a new dataset class in `data/`, inheriting from `data.BaseDataset`
2. Add a YAML config file in `data/configs/`
3. Implement `__len__`, `__getitem__`, `get_gt_json`
4. Add tests in `data/tests/test_datasets.py`

### Scenario: Modify YOLO Postprocessing (e.g., support a new YOLO version)

1. Find the corresponding code in `modelflow/processors/detect/postprocess.py`
2. YOLOv5 and YOLOv8/v11 have different output formats — differentiate via the `model_version` parameter
3. If new coordinate transformations are involved, confirm test coverage in `modelflow/tests/test_processors.py`
4. Run `pytest modelflow/tests/test_processors.py -v` to verify

### Scenario: Fix a Bug

1. First determine whether it's a spec issue or a code issue
2. If it's a spec issue: modify spec → change code → update tests
3. If it's a code issue: find the corresponding behavioral definition in the spec → change code → run tests
4. Check whether a new entry is needed in CLAUDE.md's Known Gotchas

### Scenario: Add a New Export Pipeline

1. Inherit from `BaseExporter` in `export/_base.py` and implement `export_onnx()`
2. Create a new export script in `export/onnx/`
3. Use validation methods from `export/_validation.py`
4. Add downstream conversion in `export/tensorrt/` or `export/triton/`
5. Write export tests

---

## 5. Code Review Checklist

> Universal checklist (tests, format, doc sync) is in [`SDD_METHODOLOGY.md`](SDD_METHODOLOGY.md) §3.
> The following are ModelFlow project-specific checks.

Self-check after every change:

- [ ] All 8 hard architecture constraints are not violated (see §1: zero cross-dependency between modules, zero reference / single responsibility for Backend, no direct Backend calls from Processor, Pipeline as sole orchestrator, modelflow is pure inference engine, eval is pure evaluation module, DataFlow-CV import guarded, Export does not import modelflow)
- [ ] Backend does not perform image preprocessing or postprocessing
- [ ] Preprocessor outputs the correct NCHW float32 format
- [ ] Postprocessor correctly handles empty detections (returns empty arrays when no targets)
- [ ] YOLO version parameter (`model_version`) is correctly passed through to postprocessing
- [ ] NMS `max_det`, `conf_thres`, `iou_thres` parameters are correctly passed through
- [ ] Evaluator gracefully degrades when DataFlow-CV is unavailable
- [ ] `eval/` does NOT import from `modelflow/` or `data/`
- [ ] COCO prediction format uses correct coordinate conversion (`[x1, y1, x2, y2]` → `[x, y, w, h]`)
- [ ] COCO `category_id` is 1-indexed (class_id + 1)
- [ ] New functions/classes have corresponding tests
- [ ] `pytest modelflow/tests/ eval/tests/ data/tests/ -v` all pass
- [ ] Behavior changes have been synced to specs (P0)
- [ ] New architecture details/gotchas have been synced to CLAUDE.md (P1)
- [ ] API / feature entry point changes have been synced to README.md (P1)
- [ ] User interface changes have been synced to samples/ example code (P2)

---

## 6. References

- **SDD_METHODOLOGY.md**: Universal SDD methodology (project-agnostic)
- **CLAUDE.md**: Project architecture, key details, known gotchas, development commands
- **README.md**: User documentation, installation, quick start, project structure
- **specs/evaluate/index.md**: Evaluate bridge layer overview

---

## 7. Spec Change History

| Version | Date | Changes |
|---------|------|---------|
| **0.9** | 2026-06-16 | Merged `specs/index.md` into `SDD_GUIDE.md` — single entry point for spec navigation. |
| **0.8** | 2026-06-16 | Removed `spec_cpp.md` — C++ module out of scope. Cleaned all cpp references from specs. Deleted individual changelogs from sub-specs — all change history now consolidated in SDD_GUIDE.md. Fixed Status tags: Draft → Implemented for specs with implemented code. Changelog entries rewritten to architecture-decision granularity (what changed and why, not which files). |
| **0.7** | 2026-06-16 | Added model metadata collection and latency profiling as utility modules under `utils/`. Separated model analysis (`parse_model.py`) from accuracy evaluation — eval scripts remain pure evaluation. Expanded dataset YAML config coverage. |
| **0.6** | 2026-06-16 | Specs aligned to actual code implementation. `spec_eval.md` rewritten to match constructor injection + `run()` pattern. `spec_architecture.md` updated with `utils/` and `vlms/` modules. `evaluate/` layer updated to match actual evaluator contracts. |
| **0.5** | 2026-06-15 | Redesigned `eval/` as pure evaluation module with constructor injection — zero import dependency on modelflow/ or data/. |
| **0.4** | 2026-06-15 | Added `eval/` module spec. `modelflow/` → pure inference engine. |
| **0.3** | 2026-06-14 | Split SDD_AGENT.md into SDD_METHODOLOGY.md (universal) + SDD_GUIDE.md (project-specific). Added evaluate/ WHAT layer. Added Dependency/Error contracts to module specs. |
| **0.2** | 2026-06-14 | Added evaluate/ layer. Restructured SDD_AGENT.md. |
| **0.1** | 2026-06-14 | Initial spec structure: modules/ + export/ layers. |
