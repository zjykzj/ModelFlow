# ModelFlow Export2 — 模型导出管线

独立、模块化、零外部依赖的模型导出工具，支持 **PyTorch → ONNX → TensorRT → Triton** 全链路。

## 快速开始

### L1: 导出 ONNX（最简路径）

```bash
# torchvision 分类模型
python3 -m export2.onnx.convert --model efficientnet_b0 --save model.onnx

# Ultralytics 检测模型
python3 -m export2.onnx.ultralytics yolov8s --save yolov8s.onnx
```

### L2: 导出 ONNX + TensorRT 引擎

```bash
# Step 1: PT → ONNX
python3 -m export2.onnx.convert --model efficientnet_b0 --save model.onnx

# Step 2: ONNX → TensorRT FP16
python3 -m export2.tensorrt.build_fp16 --onnx model.onnx --save model_fp16.engine

# 或 ONNX → TensorRT INT8（需校准数据）
python3 export2/scripts/generate_calib_cache_for_imagenet.py \
    --input_dir ./calib_images --output_dir ./calib_dst
python3 -m export2.tensorrt.build_int8 \
    --onnx model.onnx --calib_dir ./calib_dst --output model_int8.engine
```

### L3: 导出 ONNX + TensorRT + Triton 配置

```bash
# Step 1-2: 先导出 ONNX 和 TensorRT 引擎（同上）

# Step 3: 生成 Triton config + 部署
python3 -m export2.triton.config_generator \
    --model-name Classify_ImageNet_EfficientNetB0_TRT \
    --backend tensorrt --task classify --save ./models/triton/
```

## Python API 使用

```python
# === L1: 导出 ONNX ===
from export2.onnx import TorchvisionExporter, UltralyticsExporter

# torchvision 分类
exporter = TorchvisionExporter("efficientnet_b0", opset=12)
exporter.export_onnx("models/efficientnet_b0.onnx", img_size=224)

# Ultralytics 检测/分割/分类/姿态
exporter = UltralyticsExporter("yolov8s-seg")
exporter.export_onnx("models/yolov8s-seg.onnx")

# === L2: TensorRT FP16/INT8 ===
from export2.tensorrt import build_fp16_engine, build_int8_engine_torch

build_fp16_engine("models/yolov8s.onnx", "models/yolov8s_fp16.engine")

build_int8_engine_torch(
    onnx_path="models/yolov8s.onnx",
    calib_dir="./calib_coco/",
    output_path="models/yolov8s_int8.engine",
    input_shape=(1, 3, 640, 640),
)

# === L3: Triton 配置生成 ===
from export2.triton import TritonConfigGenerator, ModelRepoBuilder

gen = TritonConfigGenerator(
    model_name="Detect_COCO_YOLOv8s_TRT",
    backend="tensorrt",
    task="detect",
)
gen.save("models/triton/")

# 部署模型文件
builder = ModelRepoBuilder("models/triton/")
builder.deploy("Detect_COCO_YOLOv8s_TRT", "yolov8s_fp16.engine")
```

## 支持的任务与模型

| 来源 | 任务 | 模型示例 |
|------|------|---------|
| torchvision | Classification | ResNet, EfficientNet, MobileNet, ViT, ConvNeXt, DenseNet, VGG, ShuffleNet, SqueezeNet, MNASNet |
| Ultralytics | Detection | yolov8n/s/m/l/x, yolo11n/s/m/l/x |
| Ultralytics | Segmentation | yolov8n-seg/s-seg/m-seg/l-seg/x-seg |
| Ultralytics | Classification | yolov8n-cls/s-cls/m-cls/l-cls/x-cls |
| Ultralytics | Pose | yolov8n-pose/s-pose/m-pose/l-pose/x-pose |

## 导出深度等级

| 等级 | 产出 | 推理方式 | 场景 |
|------|------|---------|------|
| **L1** | `.onnx` | ONNX Runtime | 快速部署、跨平台 |
| **L2** | `.onnx` + `.engine` | TensorRT GPU | 高性能生产 |
| **L3** | `.onnx` + `.engine` + Triton 配置 | Triton Server | 服务化部署 |

## 量化策略速览

| 方式 | 精度损失 | 加速比 | 需要校准数据 |
|------|---------|:------:|:----------:|
| FP16 | ~0% | 1.5–2× | ❌ |
| INT8 | ~0.5–1% | 2–3× | ✅ 50–100 张 |

## 环境要求

- Python 3.8+
- PyTorch, torchvision, onnx, onnxruntime
- tensorrt（L2/L3 需要）
- ultralytics（Ultralytics 模型需要）
- pycuda（INT8 PyCUDA 校准器需要，可选）
- onnx-simplifier（ONNX 优化需要，可选）

## 项目结构

```
export2/
├── core/           # 基础设施（基类、验证、预处理）
├── onnx/           # PT → ONNX 导出
├── tensorrt/       # ONNX → TensorRT 引擎构建
├── triton/         # Triton 配置生成
├── scripts/        # 辅助脚本（校准数据生成等）
└── tests/          # 测试
```

## 详细文档

导出原理、格式规范、选型依据详见 [`specs/export/`](../specs/export/index.md) 知识层文档。
