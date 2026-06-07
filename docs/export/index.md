# ModelFlow Export — 模型导出管线

独立、模块化、零外部依赖的模型导出工具，支持 **PyTorch → ONNX → TensorRT → Triton** 全链路。

## 架构

```
export/
├── core/           # 基础设施（BaseExporter 基类、验证、预处理）
├── onnx/           # PT → ONNX 导出（torchvision + Ultralytics）
├── tensorrt/       # ONNX → TensorRT 引擎构建（FP16 / INT8）
├── triton/         # Triton 配置生成 + 模型仓库管理
├── scripts/        # 辅助脚本（校准数据生成）
└── tests/          # 测试
```

**关键约束：** export 模块对 `modelflow/` 零依赖，使用自包含的预处理管线 (`export/core/utils.py`)。

## 导出深度等级

| 等级 | 产出 | 推理方式 | 场景 |
|------|------|---------|------|
| **L1** | `.onnx` | ONNX Runtime (CPU/GPU) | 快速部署、跨平台 |
| **L2** | `.onnx` + `.engine` | TensorRT GPU (FP16/INT8) | 高性能生产 |
| **L3** | `.onnx` + `.engine` + Triton 配置 | Triton Server | 服务化部署 |

```mermaid
流程图参见 specs/export/index.md
```

## 模型支持矩阵

| 模型来源 | 任务 | L1 ONNX | L2 FP16 | L2 INT8 | L3 Triton |
|---------|------|:-------:|:-------:|:-------:|:---------:|
| torchvision | Classification | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Detection | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Segmentation | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Classification | ✅ | ✅ | ✅ | ✅ |
| Ultralytics | Pose | ✅ | ✅ | ✅ | ✅ |

## 快速开始

### L1: PT → ONNX

```bash
# torchvision 分类模型
python3 -m export.onnx.convert --model efficientnet_b0 --save model.onnx

# Ultralytics 检测/分割/分类/姿态模型
python3 -m export.onnx.ultralytics yolov8s --save yolov8s.onnx
python3 -m export.onnx.ultralytics yolov8s-seg --save yolov8s-seg.onnx
```

### L2: ONNX → TensorRT

```bash
# FP16 引擎（trtexec 优先，Python API 兜底）
python3 -m export.tensorrt.build_fp16 --onnx model.onnx --save model_fp16.engine

# INT8 引擎（PyTorch 校准器）
python3 -m export.tensorrt.build_int8 \
    --onnx model.onnx \
    --calib_dir ./calib_dst \
    --output model_int8.engine \
    --input_shape 1 3 640 640
```

### L3: Triton 配置生成

```bash
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt --task detect \
    --save ./models/triton/
```

## Python API

```python
# === L1: ONNX 导出 ===
from export.onnx import TorchvisionExporter, UltralyticsExporter

# torchvision 分类
exporter = TorchvisionExporter("efficientnet_b0", opset=12)
exporter.export_onnx("model.onnx", img_size=224)

# Ultralytics 检测/分割
exporter = UltralyticsExporter("yolov8s")
exporter.export_onnx("yolov8s.onnx")

# === L2: TensorRT ===
from export.tensorrt import build_fp16_engine, build_int8_engine_torch

build_fp16_engine("model.onnx", "model_fp16.engine")
build_int8_engine_torch(
    onnx_path="model.onnx",
    calib_dir="./calib_coco/",
    output_path="model_int8.engine",
    input_shape=(1, 3, 640, 640),
)

# === L3: Triton ===
from export.triton import TritonConfigGenerator, ModelRepoBuilder

gen = TritonConfigGenerator(
    model_name="Detect_COCO_YOLOv8s_TRT",
    backend="tensorrt",
    task="detect",
)
gen.save("models/triton/")

builder = ModelRepoBuilder("models/triton/")
builder.deploy("Detect_COCO_YOLOv8s_TRT", "yolov8s_fp16.engine")
```

## 按模型类型的详细指南

| 我要导出... | 看这篇 |
|-------------|--------|
| ResNet / MobileNet / EfficientNet 等分类模型 | [分类模型导出](classification.md) |
| YOLOv8 / YOLO11 / YOLO26 检测模型 | [检测模型导出](detection.md) |
| YOLOv8-seg / YOLO11-seg / YOLO26-seg 分割模型 | [分割模型导出](segmentation.md) |

## 深入指南

| 我想了解... | 看这篇 |
|-------------|--------|
| FP16 / INT8 量化原理、校准器对比、性能调优 | [TensorRT 深入指南](tensorrt.md) |
| config.pbtxt 配置、模型仓库结构、Docker 部署 | [Triton 部署指南](triton.md) |

## 环境要求

| 等级 | 依赖 |
|------|------|
| L1 基础 | Python 3.8+, PyTorch, torchvision, onnx, onnxruntime |
| L1 Ultralytics | 额外：`ultralytics` |
| L2 FP16 | 额外：`tensorrt` (`trtexec` 或 Python API) |
| L2 INT8 (PyTorch) | 额外：`tensorrt` + PyTorch CUDA |
| L2 INT8 (PyCUDA) | 额外：`tensorrt` + `pycuda`（Jetson/嵌入式） |
| L3 Triton | 额外：`tritonclient[grpc]`（客户端），Docker（服务端） |

## 量化速览

| 方式 | 精度损失 | 加速比 | 需要校准数据 |
|------|---------|:------:|:----------:|
| FP16 | ~0% | 1.5–2× | 否 |
| INT8 | ~0.5–1% | 2–3× | 是（50–100 张） |
