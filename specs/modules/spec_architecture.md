# ModelFlow 整体架构

> **Version:** 0.1
> **Status:** Draft
> **Dependencies:** None

## 1. 三层独立模块

ModelFlow 由三个平级的顶层模块组成，模块间无相互依赖：

```
ModelFlow/
├── modelflow/       # Python 推理/评估/可视化核心包
├── export/          # 模型导出（pt → onnx → tensorrt/triton）
└── cpp/             # C++ 推理（OpenCV + onnxruntime/tensorrt）
```

| 模块 | 职责 | 关键依赖 |
|------|------|----------|
| `modelflow/` | Python 推理 Pipeline、数据集加载、评估编排、可视化编排 | ONNX Runtime, TensorRT Python, Triton Client, DataFlow-CV |
| `export/` | PyTorch → ONNX → TensorRT(FP16/INT8) → Triton 配置 | PyTorch, ONNX, TensorRT |
| `cpp/` | C++ 推理 Pipeline，CPU/GPU 部署 | OpenCV, ONNX Runtime C++, TensorRT |

**关键约束**：

- Python 推理只使用 ONNX Runtime / TensorRT / Triton，**不涉及 PyTorch 推理**
- C++ 推理只使用 ONNX Runtime / TensorRT，**不涉及 PyTorch**
- PyTorch **仅**在 `export/` 模块中用于模型导出
- 评估指标和可视化**全部依赖 [DataFlow-CV](https://github.com/zjykzj/DataFlow-CV)**

## 2. 核心模式：Pipeline

```
InferencePipeline = Preprocessor + Backend + Postprocessor
```

这是贯穿三个模块的核心抽象：

```
image (HWC, BGR) → Preprocessor → tensor (NCHW) → Backend → raw → Postprocessor → result (dict)
```

| 阶段 | Python 实现 | C++ 实现 | 职责 |
|------|-------------|----------|------|
| **Preprocessor** | NumPy + OpenCV / PyTorch + TorchVision | OpenCV | 图像 → 网络输入张量 |
| **Backend** | ONNX Runtime / TensorRT / Triton | ONNX Runtime / TensorRT | 张量 → 原始推理输出 |
| **Postprocessor** | NumPy / PyTorch | 原生 C++ | 原始输出 → 结构化结果 |

## 3. 模块间关系

```
                        ┌─────────────────────┐
                        │    PyTorch (导出)     │
                        └─────────┬───────────┘
                                  │ pt → onnx
                                  ▼
                   ┌──────────────────────────┐
                   │        export/            │
                   │  onnx → tensorrt, triton  │
                   └──────────────────────────┘
                         │           │
                         │ *.onnx    │ *.engine / triton config
                         ▼           ▼
  ┌──────────────────────────────────────────┐
  │              modelflow/                   │
  │  Pipeline(processor + backend + postproc) │
  │  ┌────────┐  ┌────────┐  ┌────────────┐  │
  │  │onnx.py │  │trt.py  │  │triton.py   │  │
  │  └────────┘  └────────┘  └────────────┘  │
  │                                          │
  │  Evaluator ──▶ DataFlow-CV (metrics)     │
  │  Visualizer ─▶ DataFlow-CV (drawing)     │
  └──────────────────────────────────────────┘

  ┌──────────────────────────────────────────┐
  │              cpp/                         │
  │  core/(OpenCV) + backends/onnx/  ── CPU   │
  │  core/(OpenCV) + backends/tensorrt/ ── GPU│
  │  (各后端可独立提取)                        │
  └──────────────────────────────────────────┘
```

## 4. 数据流

### 4.1 推理数据流

```
用户输入（图像路径 / cv2 image）
        │
        ▼
Preprocessor.__call__(image)
    ├── NumPy:  letterbox/resize → HWC→CHW → normalize
    └── PyTorch: torchvision.transforms
        │
        ▼  Input Tensor (NCHW, float32)
Backend.__call__(tensor)
    ├── ONNX:   session.run()
    ├── TensorRT: execute_async_v2()
    └── Triton:  grpc/http infer()
        │
        ▼  Raw Outputs (List[np.ndarray])
Postprocessor.__call__(raw, **kwargs)
    ├── classify:  softmax → top-k
    ├── detect:    NMS → scale_boxes → (boxes, scores, labels)
    ├── segment:   NMS → process_mask → (boxes, scores, labels, masks)
    ├── semantic:  argmax → colormap
    └── multimodal: similarity → prob
        │
        ▼  Structured Result (dict)
```

### 4.2 评估数据流

```
Dataset ──逐张──▶ Pipeline(image) ──▶ 收集结果 ──▶ 转换为 COCO JSON
                                                        │
                                                        ▼
                                              DataFlow-CV Evaluator
                                                        │
                                                        ▼
                                              Metrics dict (mAP, AP50...)
```

### 4.3 可视化数据流

```
Pipeline result + Dataset ──▶ COCO JSON ──▶ DataFlow-CV Visualizer
                                                    │
                                                    ▼
                                          Annotated images
```

## 5. 任务覆盖矩阵

| 任务类型 | Python Pipeline | C++ Pipeline | Export 导出 | Evaluator | Visualizer |
|----------|:---------------:|:------------:|:-----------:|:---------:|:----------:|
| Classification | ✅ | ✅ | ✅ | 本地实现 | 本地实现 |
| Detection | ✅ | ✅ | ✅ | DataFlow-CV | DataFlow-CV |
| Instance Segmentation | ✅ | ✅ | ✅ | DataFlow-CV | DataFlow-CV |
| Semantic Segmentation | ✅ | ❌ | ✅ | 本地实现 | 本地实现 |
| Multi-modal (CLIP) | ✅ | ❌ | ✅ | 本地实现 | 本地实现 |

## 6. 实施路线

| Phase | 内容 | 交付物 |
|-------|------|--------|
| 1 | 基础设施 | interfaces, registry, types, Pipeline 核心 |
| 2 | 分类推理+评估 | OnnxBackend, ClassifyProcessor, 分类 Evaluator |
| 3 | 检测推理+评估 | TensorrtBackend, TritonBackend, DetectProcessor, DataFlow-CV 桥接 |
| 4 | 分割+语义分割 | SegmentProcessor, SemanticSegProcessor |
| 5 | 多模态 | CLIP Processor |
| 6 | Export 模块 | pt→onnx, onnx→trt, triton config |
| 7 | C++ 模块 | core, backends/onnx, backends/tensorrt, examples |
