# Changelog

All notable changes to ModelFlow are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/),
and this project uses [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-06-07

### Added

- **SDD Agent 开发方法论**：`specs/SDD_AGENT.md` — 面向 AI Agent 的规范驱动开发指南
  - 三层体系（Specs → CLAUDE.md → 代码）和开发铁律
  - 完整开发工作流：确定影响范围 → 按模块读 spec → 对照 CLAUDE.md → 写代码 → 提交前检查
  - 架构硬约束（5 条）、Backend/Processor/Evaluator 契约速查
  - Specs 导航地图、常见开发场景速查、代码审查清单
- **CLAUDE.md 增强**：从 DataFlow-CV 引入成熟模式
  - Specifications 章节（specs vs CLAUDE.md 职责划分）
  - 架构约束图 + 5 条硬约束
  - Critical Implementation Details（Backend 契约、YOLO 版本差异、Mask 解码、DataFlow-CV 桥接）
  - Known Gotchas（12 条常见陷阱）
  - Test Structure 文档与 Git 提交规范化

### Changed

- **CLAUDE.md**：451 行完整重写，增加 250+ 行关键上下文
- **版本提升**：0.3.0 → 0.4.0

## [0.3.0] - 2026-06-07

### Added

- **modelflow/ Python 核心包**：完整的推理/评估/可视化框架
  - `core/` — 抽象基类（InferencePipeline, BaseBackend, BasePreprocessor/Postprocessor, BaseDataset, BaseMetrics, BaseEvaluator, BaseVisualizer）、Registry 注册机制、类型枚举、ModelConfig
  - `backends/` — OnnxBackend / TensorrtBackend / TritonBackend 统一接口
  - `processors/` — classify（ImageNet 标准预处理、softmax top-k）、detect（letterbox、NMS，支持 v5/v8/v11）、segment（proto mask 解码）、semantic_seg（argmax + colormap）、multimodal（CLIP 图像/文本预处理 + 相似度排序）
  - `pipelines/` — 4 个 Pipeline 工厂函数（classify/detect/segment/semantic_seg）
  - `datasets/` — COCODetectionDataset、COCOSegmentDataset、ClassifyDataset
  - `evaluators/` — DetectEvaluator（DataFlow-CV 桥接 mAP）、ClassifyEvaluator（本地混淆矩阵）、SegmentEvaluator
  - `metrics/` — ClassificationMetrics（混淆矩阵 → Accuracy/Precision/Recall/F1）
  - `viz/` — DetectVisualizer（OpenCV 绘制 + DataFlow-CV 桥接）
  - `cfgs/` — COCO 80 类、ImageNet 1000 类
- **Export 模块重构**：`export/` 重写为 23 文件模块化结构
  - `export/core/` — BaseExporter 基类、validation、自包含预处理 utils
  - `export/onnx/` — TorchvisionExporter（40+ 模型）、UltralyticsExporter（YOLOv8/v11+）、ONNX optimize
  - `export/tensorrt/` — FP16 构建器（trtexec + Python API）、INT8 构建器（PyTorch/PyCUDA 分层校准器）
  - `export/triton/` — config.pbtxt 自动生成、模型仓库管理
  - `export/scripts/` — 校准数据生成脚本（零外部依赖）
  - `export/tests/` — 单元测试 + 集成测试
- **Specs 文档系统**：`specs/` 完整的分层架构设计
  - `specs/modules/` — 架构设计文档（architecture / python / export / cpp）
  - `specs/export/` — 知识层文档（ONNX 导出规范、TensorRT 转换原理、Triton 部署配置）

### Changed

- **eval/ 基准测试改造**：13 个 bench 脚本全部改为调用 modelflow，取消对 legacy core/ 的依赖
- **文档更新**：README.md 重写、CLAUDE.md 更新为新架构
- **版本提升**：0.2.0 → 0.3.0

### Removed

- **export/ 旧实现**：移除扁平结构（pytorch_to_onnx.py, ultralytics_export.py, safe_int8_build_*.py）

## [0.2.0] - 2026-06-07

### Added

- **Architecture Specs**: 完整的分层架构设计文档 `specs/modules/`
  - `spec_architecture.md` — 三层独立模块设计（modelflow / export / cpp）
  - `spec_python.md` — Python 推理/评估框架接口设计
  - `spec_export.md` — 模型导出管线规范（pt → onnx → tensorrt/triton）
  - `spec_cpp.md` — C++ 推理模块设计（独立可提取后端）
- **CLAUDE.md**: 项目 AI 辅助开发指南

### Changed

- **Backends 重构**：废弃旧版 `BackendRuntime` / `BackendTensorRT`，引入统一 `ONNXModel` / `TRTModel` / `TritonModel` 接口，共享 `IOInfo` dataclass
- **Eval 评估器重构**：分类/检测/分割评估器统一 `run()` + `eval()` 模式，支持 COCO mAP 计算
- **Preprocessor 统一**：分类预处理支持 resize 和 crop 两种模式，检测预处理统一 letterbox
- **CLIP 评估优化**：CIFAR-10/CIFAR-100 零样本和 Linear Probe 评估，支持集成模板策略
- **gitignore 更新**：新增 *.onnx, *.engine, *.plan, *.trt 等模型文件忽略规则

### Added (since v0.1.0)

- **TensorRT 后端**：`TRTModel` 类，支持 FP16/INT8 引擎加载、CUDA buffer 管理、异步推理
- **Triton 后端**：`TritonModel` 类，支持 gRPC/HTTP 协议、model metadata 查询、health check
- **YOLOv8 检测推理**：numpy 和 torch 两种预处理/后处理实现
- **YOLOv8-seg 实例分割推理**：完整 mask 解码/裁剪 Pipeline
- **YOLOv5/v8/v8-seg Triton 推理**：三个模型在 Triton Inference Server 上的推理实现
- **COCO 评估基准**：YOLOv5s（~36.5 mAP）、YOLOv8s（~43.7 mAP）、YOLOv8s-seg（bbox ~42.8%, segm ~33.6%）
- **TensorRT INT8 量化**：`safe_int8_build_by_torch.py` 和 `safe_int8_build_by_pycuda.py` 两种校准器
- **校准缓存生成**：COCO 和 ImageNet 两个校准数据集的 `.bin` 缓存生成脚本
- **ImageNet 分类评估**：EfficientNet-B0 在 50K 验证集上的精度基准（FP16 ~77.68%, INT8 ~72.18%）
- **Triton 模型仓库结构**：6 个预配置的 `config.pbtxt` 模板
- **OpenCLIP 支持**：Linear Probe + KNN 分类器评估
- **SAM3 示例**：`sam3_sample.py` 文本提示驱动的分割演示

## [0.1.0] - 2025-09-06

### Added

- 项目初始化
- YOLOv5 ONNX Runtime 推理（NumPy 预处理/后处理）
- YOLOv5 PyTorch 预处理实现
- 基本数据加载器（`LoadImages` 支持图像/视频/流）
- 基础后端抽象 `BackendRuntime` / `BackendTensorRT`
- 标注工具类：YOLOv5 Annotator、Colors
- NumPy NMS 实现（YOLOv5 格式）
- 示例图片和 COCO128 获取脚本
- 项目 Logo
- 基础 README 文档
