# Changelog

All notable changes to ModelFlow are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/),
and this project uses [Semantic Versioning](https://semver.org/).

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
