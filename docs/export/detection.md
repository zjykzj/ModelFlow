# 检测模型全链路导出

> 适用于 Ultralytics YOLO 检测模型：YOLOv8、YOLO11、YOLO26。

## 支持的模型

| 系列 | 模型 | 输入尺寸 | 输出结构 | 官方权重 |
|------|------|:-------:|------|:-------:|
| **YOLOv8** | yolov8n / s / m / l / x | 640×640 | `(1, 84, 8400)` | COCO |
| **YOLO11** | yolo11n / s / m / l / x | 640×640 | `(1, 84, 8400)` | COCO |
| **YOLO26** | yolo26n / s / m / l / x | 640×640 | `(1, 84, 8400)` | COCO |

> **兼容性说明：** YOLOv8 → YOLO11 → YOLO26 的输出结构完全兼容（`84 = 4 bbox + 80 COCO classes`, `8400 = 80×80 + 40×40 + 20×20` 检测头 grid 点），后处理无需修改。

## 全链路总览

```
Ultralytics YOLO 模型 (.pt)
    │
    ├── L1 ──▶ export.onnx.ultralytics ──▶ .onnx ──▶ ONNX Runtime / TensorRT / Triton
    │
    ├── L2 ──▶ export.tensorrt.build_fp16 ──▶ .engine (FP16)
    │   └──▶ export.tensorrt.build_int8 ──▶ .engine (INT8) ← 需校准数据
    │
    └── L3 ──▶ export.triton.config_generator ──▶ Triton 部署
```

---

## L1: PT → ONNX

### 导出器：`UltralyticsExporter`

位于 `export/onnx/ultralytics.py`，底层调用 Ultralytics 的 `YOLO().export(format="onnx")`，封装了 opset 策略和文件管理。

**导出流程：**

1. 自动识别任务类型（`-seg` → segment, `-cls` → classify, `-pose` → pose，否则 detect）
2. 自动选择次新版 opset（`max_opset - 1`，确保稳定性）
3. `YOLO(model_name).export(format="onnx", opset=..., imgsz=...)`
4. 将生成的 `.onnx` 移动到目标路径

### CLI 参数

```
python3 -m export.onnx.ultralytics --help
```

| 参数 | 必填 | 默认 | 说明 |
|------|:---:|------|------|
| `model` (位置) | ✅ | — | 模型名（`yolov8s`、`yolo11n`） |
| `--opset` | — | 次新版 | ONNX opset 版本 |
| `--save` | — | `<model>.onnx` | ONNX 保存路径 |
| `--img-size` | — | `640` | 输入尺寸 |

### 使用示例

```bash
# YOLOv8 系列
python3 -m export.onnx.ultralytics yolov8n --save models/runtime/yolov8n.onnx
python3 -m export.onnx.ultralytics yolov8s --save models/runtime/yolov8s.onnx
python3 -m export.onnx.ultralytics yolov8m --save models/runtime/yolov8m.onnx
python3 -m export.onnx.ultralytics yolov8l --save models/runtime/yolov8l.onnx
python3 -m export.onnx.ultralytics yolov8x --save models/runtime/yolov8x.onnx

# YOLO11 系列
python3 -m export.onnx.ultralytics yolo11n --save models/runtime/yolo11n.onnx
python3 -m export.onnx.ultralytics yolo11s --save models/runtime/yolo11s.onnx

# YOLO26 系列（未来版本，接口一致）
python3 -m export.onnx.ultralytics yolo26s --save models/runtime/yolo26s.onnx

# 指定 opset 版本
python3 -m export.onnx.ultralytics yolov8s --save yolov8s.onnx --opset 17
```

### Python API

```python
from export.onnx import UltralyticsExporter

# 基础用法
exporter = UltralyticsExporter("yolov8s")
onnx_path = exporter.export_onnx("models/runtime/yolov8s.onnx")

# 自定义 opset（不传则自动选择次新版）
exporter = UltralyticsExporter("yolo11m", opset=17)
onnx_path = exporter.export_onnx(
    output_path="models/runtime/yolo11m.onnx",
    img_size=640,
)
```

### Opset 版本策略

`UltralyticsExporter` 默认使用 `torch.onnx` 支持的 **次新版 opset**（`max_opset - 1`），而非最新版。原因：
- 最新 opset 可能存在未修复的 bug
- 次新版经过更长时间的验证，更稳定
- TensorRT 对特定 opset 版本有最佳兼容性

### 输入输出规格

| 项目 | 规范 |
|------|------|
| 输入名 | `image` |
| 输入 shape | `(1, 3, 640, 640)` |
| 输入 dtype | `float32` |
| 输入范围 | `[0, 1]`（LetterBox + ÷255 后） |
| 输出名 | `output0` |
| 输出 shape (detect) | `(1, 84, 8400)` |
| 输出含义 | 检测头原始输出（4 bbox + 80 classes + 未后处理） |

---

## L2: ONNX → TensorRT

### FP16 引擎

```bash
python3 -m export.tensorrt.build_fp16 \
    --onnx models/runtime/yolov8s.onnx \
    --save models/tensorrt/yolov8s_fp16.engine \
    --workspace 4
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--onnx` | (必填) | ONNX 路径 |
| `--save` | (必填) | 引擎路径 |
| `--workspace` | `4` | 工作空间 GB |
| `--no-trtexec` | `false` | 跳过 trtexec |
| `--shapes` | — | 静态 shape `input:1x3x640x640` |

### INT8 引擎（PyTorch 校准器）

```bash
# Step 1: 生成校准数据（COCO 子集，50-100 张）
python3 export/scripts/generate_calib_cache_for_coco.py \
    --input_dir ./cal_coco_src \
    --output_dir ./cal_coco_dst \
    --input_size 640 \
    --max_images 100

# Step 2: 构建 INT8 引擎
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir ./cal_coco_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640 \
    --workspace 4
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--onnx` | (必填) | ONNX 路径 |
| `--calib_dir` | (必填) | 校准 `.bin` 目录 |
| `--output` | `model_int8.engine` | 保存路径 |
| `--input_shape` | `1 3 640 640` | N C H W |
| `--workspace` | `4` | 工作空间 GB |
| `--cache_file` | `calib_torch.cache` | 缓存文件名 |

### 校准数据生成

```bash
python3 export/scripts/generate_calib_cache_for_coco.py \
    --input_dir /path/to/coco_val2017 \
    --output_dir ./calib_dst \
    --input_size 640 \
    --max_images 100 \
    --format bin
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input_dir` | (必填) | COCO 图片目录 |
| `--output_dir` | (必填) | 输出 `.bin` 目录 |
| `--input_size` | `640` | 目标尺寸 |
| `--format` | `bin` | `bin` / `npy` |
| `--max_images` | `100` | 最大处理数 |

**预处理管线：** `LetterBox(640) → BGR→RGB → /255 → CHW float32`

**输出文件大小：** `640 × 640 × 3 × 4 = 4,915,200 Bytes (~4.7 MB)`

---

## L3: Triton 配置

### config.pbtxt 生成

```bash
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt \
    --task detect \
    --save ./models/triton/
```

### 部署模型文件

```python
from export.triton import ModelRepoBuilder

builder = ModelRepoBuilder("models/triton/")

# 方式 1: 自动推断后端（根据文件后缀）
builder.deploy("Detect_COCO_YOLOv8s_TRT", "yolov8s_fp16.engine")
builder.deploy("Detect_COCO_YOLOv8s_ONNX", "yolov8s.onnx")

# 方式 2: 显式指定后端
builder.deploy(
    model_name="Detect_COCO_YOLOv8s_TRT",
    model_file="models/tensorrt/yolov8s_fp16.engine",
    backend="tensorrt",
)
```

### 生成的目录结构

```
models/triton/
└── Detect_COCO_YOLOv8s_TRT/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

### 生成的 config.pbtxt

```protobuf
name: "Detect_COCO_YOLOv8s_TRT"
platform: "tensorrt_plan"
max_batch_size: 0

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [84, 8400]
  }
]
```

### 多模型部署示例

```bash
# ONNX 后端
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_ONNX \
    --backend onnxruntime --task detect --save ./models/triton/

# TensorRT FP16 后端
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt --task detect --save ./models/triton/

# TensorRT INT8 后端
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_INT8 \
    --backend tensorrt --task detect --save ./models/triton/
```

---

## 输出结构详解

### 检测输出 `(1, 84, 8400)`

```
pred[0, :, i] = [cx, cy, w, h, objectness, cls_0, cls_1, ..., cls_79]
                 ├── 4 bbox params (cx, cy, w, h) — 相对于 grid cell
                 ├── 1 objectness score
                 └── 80 class scores (COCO classes)
```

| 维度 | 值 | 含义 |
|------|------|------|
| Batch | 1 | 单张图片 |
| Channels | 84 | 4 bbox + 80 classes（YOLOv8 无独立 objectness） |
| Grid | 8400 | 80×80 + 40×40 + 20×20 三检测头 |

### 后处理要点

- **坐标解码：** `cx, cy` 是相对于 grid cell 的偏移，需加 grid 坐标后归一化
- **NMS：** 完成坐标缩放（逆 LetterBox）后执行
- **模型版本无关：** v8/v11/v26 后处理完全一致

---

## ONNX 优化（可选）

```python
from export.onnx import optimize_onnx
optimize_onnx("yolov8s.onnx", "yolov8s_simplified.onnx")
```

---

## 深入阅读

- [TensorRT 深入指南](tensorrt.md) — FP16/INT8 量化原理、双校准器对比、性能调优
- [Triton 部署指南](triton.md) — config.pbtxt 完整配置、dynamic batching、Docker 部署
- [`specs/export/onnx_export.md`](../specs/export/onnx_export.md) — Ultralytics 导出规范
