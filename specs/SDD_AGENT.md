# SDD Agent 开发指南

> **本文档定义 ModelFlow 项目的 SDD Agent 开发方法论。**
>
> 目标读者：接手本项目的 AI Agent（如 Claude Code）。也适用于人类开发者。

---

## 一、核心理念

```
Specs（什么是对的）→ CLAUDE.md（代码怎么写）→ 制定计划 → 代码实现
     ↑         ← （行为变化时回写）                       |
     └─────────── 测试验证 + 文档同步 ←───────────────────┘
```

**SDD（Spec-Driven Development）Agent 开发的三层体系：**

| 层级 | 文件 | 角色 | 修改频率 |
|------|------|------|----------|
| **Specs** | `specs/` | 行为契约——定义"什么是对的" | 很少（需求变更时才改） |
| **CLAUDE.md** | `CLAUDE.md` | 开发上下文——描述"代码怎么写的" | 随代码演进 |
| **Code** | `modelflow/` `export/` `samples/` | 实现——实际运行的代码 | 日常 |

**开发铁律**：
- Specs 是最高权威。如果代码行为与 specs 冲突，以 specs 为准，改代码。
- Specs 是活文档。需求变更或 specs 不充分时，优先更新 specs，再动手。

---

## 二、开发工作流

### 2.1 接到新任务时

**第一步：确定影响范围**

问自己三个问题：
1. 改动涉及哪个大模块？（`modelflow/` / `export/` / `samples/`）
2. 涉及哪个 Pipeline 阶段？（Preprocessor / Backend / Postprocessor）
3. 涉及哪个任务类型？（Classify / Detect / Segment / SemanticSeg / Multimodal）

**第二步：读 spec，评估充分性（按此顺序）**

> ⚠️ **Specs 是活文档。** 读 spec 时带着批判眼光：
> - Specs 是否覆盖了当前场景？定义是否清晰无歧义？
> - Specs 的行为定义是否合理、内部是否一致？
> - **如果不充分或不合理 → 优先更新 specs，再往下走。** 不在不稳固的基础上盖楼。

按改动类型找对应 spec：

| 改动类型 | 必读 spec |
|----------|-----------|
| 新增/修改推理后端 | `specs/modules/spec_python.md`（第 4 节：Backend） |
| 新增/修改预处理/后处理 | `specs/modules/spec_python.md`（第 5 节：Processor） |
| 新增/修改 Pipeline | `specs/modules/spec_python.md`（第 3 节：InferencePipeline） |
| 新增/修改评估器 | `specs/modules/spec_python.md`（第 6 节：Evaluator） |
| 新增/修改数据集 | `specs/modules/spec_python.md`（第 7 节：Dataset） |
| 新增 ONNX 导出 | `specs/modules/spec_export.md` → `specs/export/onnx_export.md` |
| 新增 TensorRT 导出 | `specs/modules/spec_export.md` → `specs/export/tensorrt_conversion.md` |
| 新增 Triton 部署 | `specs/modules/spec_export.md` → `specs/export/triton_deployment.md` |
| 跨模块改动 | `specs/modules/spec_architecture.md`（架构约束图 + 模块依赖规则） |
| 不确定影响范围 | `specs/modules/spec_architecture.md` 和 `specs/index.md`（全局概览） |

**第三步：对照 CLAUDE.md**

读 CLAUDE.md 的相关章节，重点关注：
- **Known Gotchas**（常见陷阱）
- **Critical Implementation Details**（Backend 契约、Processor 数据流）

**第四步：制定开发计划**

读完 specs（知道"什么是对的"）和 CLAUDE.md（知道"代码怎么写的"）之后，**显式制定开发计划**再写代码：

1. **列出涉及的所有文件**：哪些需要创建、修改、删除
2. **确定改动顺序**：按依赖关系排定先后（先改基础接口，再改上层调用）
3. **识别风险点**：可能影响哪些现有功能？哪个环节最容易出错？
4. **复杂任务用 Plan**：跨模块改动或新增任务类型时，使用 `EnterPlanMode` / `Plan` agent 生成完整方案

> 先计划，再实现——避免在实现中途发现架构冲突、推倒重来。

### 2.2 动手写代码时

**架构硬约束（写代码前默念一遍）**

| # | 约束 | 违反后果 |
|---|------|----------|
| 1 | **`modelflow/` ↔ `export/`：零交叉依赖** | 模块耦合，无法独立开发 |
| 2 | **Backend → Preprocessor/Postprocessor：零引用** | 后端不应感知图像预处理 |
| 3 | **Backend 只接收 `np.ndarray`，只返回 `List[np.ndarray]`** | 破坏 Pipeline 抽象 |
| 4 | **Processor → Backend：不直接调用** | Pipeline 是唯一编排者 |
| 5 | **Evaluator → Pipeline：仅通过公开接口** | 绕过 pipeline 直接访问 backend |
| 6 | **Export → modelflow：零导入** | export 使用自包含的 `export/core/utils.py` 预处理 |

**Backend 契约（最容易出错的地方）**

```
输入：np.ndarray — NCHW float32，已经过 Preprocessor 预处理
输出：List[np.ndarray] — 模型原始输出列表

❌ Backend 内部不应做任何图像处理（resize/normalize/letterbox）
❌ Backend 内部不应做任何后处理（NMS/softmax/解码）
✅ Backend 只做单一职责：执行推理，返回原始输出
```

**Preprocessor 数据流**

```
输入：np.ndarray — HWC uint8 BGR (OpenCV 标准加载格式)

classify:  BGR→RGB → Resize(256) → CenterCrop(224) → Normalize → HWC→CHW → BatchDim
detect:    LetterBox → BGR→RGB → /255 → HWC→CHW
segment:   (同 detect)
semantic_seg: BGR→RGB → Resize(可选) → /255 → Normalize → HWC→CHW
multimodal:   BGR→RGB → Resize(短边224) → CenterCrop(224) → CLIP Normalize → HWC→CHW
```

**Postprocessor 数据流**

```
输入：List[np.ndarray] — 模型原始输出列表

classify:  softmax → top-k → {class_ids, scores, class_names?}
detect:    转置 → 阈值过滤 → xywh2xyxy → NMS → scale_boxes → {boxes, scores, class_ids}
segment:   阈值过滤 → NMS → process_mask(proto + coeffs) → crop_mask → {boxes, scores, class_ids, masks}
semantic_seg: argmax → colormap(可选) → {class_map, colormap?}
multimodal:   embedding归一化 → 相似度 → softmax → top-k → {similarity, probs, indices, labels}
```

**Pipeline 调用约定**

```python
pipeline = InferencePipeline(preprocessor, backend, postprocessor)

# 端到端推理（推荐）
result = pipeline(image, conf_thres=0.25, iou_thres=0.45)

# 跳过预处理/后处理（评估用）
raw_outputs = pipeline.infer(tensor)
```

**Evaluator 流**

```
ClassifyEvaluator: pipeline(image) → ClassificationMetrics.update() → metrics.compute()
DetectEvaluator:   pipeline(image) → 收集 COCO 格式预测 → DataFlow-CV DetectionEvaluator → mAP
SegmentEvaluator:  pipeline(image) → 收集 COCO 格式预测(含 mask RLE) → DataFlow-CV DetectionEvaluator → mAP
```

**Export 深度等级**

| 等级 | 产出 | 运行方式 |
|------|------|---------|
| **L1** | `.onnx` | ONNX Runtime (CPU/GPU) |
| **L2** | `.onnx` + `.engine` | TensorRT GPU (FP16/INT8) |
| **L3** | `.onnx` + `.engine` + Triton config | Triton Inference Server |

```
Export 数据流：
PT Model → BaseExporter.export_onnx() → .onnx → TensorRT Builder → .engine → Triton 配置 → model repo
              ↓
        validation.check_onnx() + validation.compare_torch_onnx()
```

### 2.3 提交前

```bash
# 1. 跑测试（必须通过）
pytest tests/ -v

# 2. 如果改了 export 相关代码，跑 export 测试
pytest export/tests/ -v
```

**文档同步检查（每次改动必做）：**

| 优先级 | 文档 | 检查条件 | 操作 |
|--------|------|---------|------|
| **P0** | `specs/` | 行为发生变化（接口、契约、数据流） | **必须**同步更新 |
| **P1** | `CLAUDE.md` | 新增架构细节、新陷阱、新硬约束、新关键实现 | 更新 Known Gotchas 或 Critical Details |
| **P1** | `README.md` | API 变化、新增功能入口、安装步骤变化 | 同步更新用户文档 |
| **P2** | `samples/` | 用户 API 变化、新增 task 类型、调用方式变化 | 更新示例代码（`samples/infer.py` 等） |

**Git commit 格式：**

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body if needed>

Co-Authored-By: DeepSeek-V4.0 <noreply@deepseek.com>
EOF
)"
```

类型：`feat` / `fix` / `docs` / `refactor` / `test` / `style` / `perf` / `chore`

---

## 三、Specs 导航地图

### 3.1 我要找什么？

```
"Pipeline 的调用流程是什么？"
  → specs/modules/spec_python.md（第 3 节：InferencePipeline）

"Preprocessor 的输出是什么格式？"
  → specs/modules/spec_python.md（第 5.2 节：Processor Input/Output Specification）

"Backend 的 __call__ 契约是什么？"
  → specs/modules/spec_python.md（第 4.2 节：Backend Interface）

"NMS 是怎么实现的？"
  → specs/modules/spec_python.md（第 5.3 节：Processors → DetectPostprocessor）

"Mask 解码是怎么做的？"
  → specs/modules/spec_python.md（第 5.3 节：Processors → SegmentPostprocessor）

"怎么加一个新的后端？"
  → specs/modules/spec_python.md（第 4.3 节：Registering a New Backend）

"评估结果的 metrics 有哪些？"
  → specs/modules/spec_python.md（第 6 节：Evaluator）

"ONNX 导出的步骤是什么？"
  → specs/modules/spec_export.md（第 3 节：Export Pipeline）

"TensorRT 怎么选 FP16 还是 INT8？"
  → specs/modules/spec_export.md（第 4 节：Conversion Strategy）
  → specs/export/tensorrt_conversion.md（量化决策树）

"Triton 模型仓库的目录结构是什么？"
  → specs/export/triton_deployment.md（第 2 节：Model Repository Structure）

"各模块之间的依赖关系是怎样的？"
  → specs/modules/spec_architecture.md（Architecture Constraint 图）

"C++ 后端的精度怎么和 Python 对齐？"
  → specs/modules/spec_cpp.md（精度对齐测试）
```

### 3.2 文件清单

```
specs/
├── SDD_AGENT.md                  # 本文档 — SDD Agent 开发指南
│
├── modules/                      # HOW — 内部模块架构
│   ├── index.md                  # Modules 层概览
│   ├── spec_architecture.md      # 架构总览：三模块 + Pipeline 模式
│   ├── spec_python.md            # Python 包：ABCs、后端、处理器、评估器、注册表
│   ├── spec_export.md            # 导出模块：Pipeline、深度等级、精度验证
│   └── spec_cpp.md               # C++ 推理模块（规划中）
│
├── export/                       # WHAT — 导出格式与转换知识
│   ├── index.md                  # 导出知识层概览
│   ├── onnx_export.md            # ONNX 导出原理
│   ├── tensorrt_conversion.md    # TensorRT 转换与量化
│   └── triton_deployment.md      # Triton 部署配置
│
└── index.md                      # 体系结构总入口
```

---

## 四、常见开发场景速查

### 场景：新增一个推理后端

1. 在 `specs/modules/spec_python.md` 确认 Backend Interface 契约
2. 在 `modelflow/backends/` 新建 `your_backend.py`，继承 `BaseBackend`
3. 用 `@BACKENDS.register("your_backend")` 注册
4. 在 `modelflow/core/types.py` 的 `BackendType` 新增枚举值
5. 在 `modelflow/pipelines/*.py` 的工厂函数新增 backend 分支
6. 在 `tests/test_backends.py` 加测试
7. 在 `tests/test_pipelines.py` 加 pipeline 测试

### 场景：新增一个任务类型（如 pose）

1. 在 `modelflow/core/types.py` 的 `TaskType` 新增枚举
2. 在 `modelflow/processors/` 新建 `pose/` 子包（preprocess.py + postprocess.py）
3. 用 `@PROCESSORS.register("pose_preprocess_npy")` 注册
4. 在 `modelflow/pipelines/` 新建 `pose.py` 工厂函数
5. 在 `modelflow/evaluators/` 新增对应 evaluator
6. 在 `samples/infer.py` 和 `samples/eval_bench.py` 新增 task 分支
7. 在 `export/` 确认导出管线支持
8. 写测试

### 场景：新增一个数据集

1. 在 `modelflow/datasets/` 新建数据集类，继承 `BaseDataset`
2. 用 `@DATASETS.register("your_dataset")` 注册
3. 实现 `__len__`、`__getitem__`、`get_gt_json`
4. 在 `tests/test_datasets.py` 加测试

### 场景：修改 YOLO 后处理（如支持新的 YOLO 版本）

1. 在 `modelflow/processors/detect/postprocess.py` 找到对应代码
2. YOLOv5 和 YOLOv8/v11 的输出格式不同，通过 `model_version` 参数区分
3. 如果涉及新的坐标变换，确认 `tests/test_processors.py` 中的测试覆盖
4. 运行 `pytest tests/test_processors.py -v` 验证

### 场景：修复 Bug

1. 先确认是 spec 问题还是代码问题
2. 如果是 spec 问题：修改 spec → 改代码 → 更新测试
3. 如果是代码问题：在 spec 中找到对应的行为定义 → 改代码 → 跑测试
4. 检查 CLAUDE.md 的 Known Gotchas 是否需要新增条目

### 场景：新增导出管线

1. 在 `export/core/base.py` 继承 `BaseExporter` 实现 `export_onnx()`
2. 在 `export/onnx/` 新建导出脚本
3. 实现 `export/core/validation.py` 中的校验
4. 在 `export/tensorrt/` 或 `export/triton/` 添加后续转换
5. 写导出测试

---

## 五、代码审查检查清单

每次改动后自查：

- [ ] 6 条架构硬约束未被违反（详见 2.2 节：模块零交叉依赖、Backend 零引用/单一职责、Processor 不直调 Backend、Pipeline 为唯一编排者、Evaluator 仅用公开接口、Export 不导入 modelflow）
- [ ] Backend 未做图像预处理或后处理
- [ ] Preprocessor 输出正确的 NCHW float32 格式
- [ ] Postprocessor 正确处理空检测（无目标时返回空数组）
- [ ] YOLO 版本参数 (`model_version`) 正确透传到后处理
- [ ] NMS 的 `max_det`、`conf_thres`、`iou_thres` 正确透传
- [ ] Evaluator 在 DataFlow-CV 不可用时优雅降级
- [ ] 新增函数/类有对应的测试
- [ ] `pytest tests/ -v` 全部通过
- [ ] 行为变化已同步更新 specs（P0）
- [ ] 新增架构细节/陷阱已同步更新 CLAUDE.md（P1）
- [ ] API / 功能入口变化已同步更新 README.md（P1）
- [ ] 用户接口变化已同步更新 samples/ 示例代码（P2）

---

## 六、参考

- **CLAUDE.md**：项目架构、关键细节、已知陷阱、开发命令
- **README.md**：用户文档、安装、快速开始、项目结构
- **specs/export/index.md**：导出深度等级和模型支持矩阵
