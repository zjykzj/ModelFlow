# 分割模型全链路导出

> 适用于 Ultralytics YOLO 实例分割模型：YOLOv8-seg、YOLO11-seg、YOLO26-seg。

## 支持的模型

| 系列 | 模型 | 输入尺寸 | 输出结构 | 官方权重 |
|------|------|:-------:|------|:-------:|
| **YOLOv8-Seg** | yolov8n-seg / s-seg / m-seg / l-seg / x-seg | 640×640 | `(1, 116, 8400)` + `(1, 32, 160, 160)` | COCO |
| **YOLO11-Seg** | yolo11n-seg / s-seg / m-seg / l-seg / x-seg | 640×640 | 同上 | COCO |
| **YOLO26-Seg** | yolo26n-seg / s-seg / m-seg / l-seg / x-seg | 640×640 | 同上 | COCO |

> **分割输出 = 检测输出 + 掩码系数 + 原型掩码。** `116 = 4 bbox + 80 COCO classes + 32 mask coeffs`。

## 全链路总览

```
Ultralytics YOLO-Seg 模型 (.pt)
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

### CLI 使用

分割模型的导出使用与检测模型相同的 `UltralyticsExporter`，任务类型由模型名后缀 `-seg` 自动识别。

```bash
# YOLOv8 分割系列
python3 -m export.onnx.ultralytics yolov8n-seg --save models/runtime/yolov8n-seg.onnx
python3 -m export.onnx.ultralytics yolov8s-seg --save models/runtime/yolov8s-seg.onnx
python3 -m export.onnx.ultralytics yolov8m-seg --save models/runtime/yolov8m-seg.onnx
python3 -m export.onnx.ultralytics yolov8l-seg --save models/runtime/yolov8l-seg.onnx
python3 -m export.onnx.ultralytics yolov8x-seg --save models/runtime/yolov8x-seg.onnx

# YOLO11 分割系列
python3 -m export.onnx.ultralytics yolo11s-seg --save models/runtime/yolo11s-seg.onnx

# YOLO26 分割系列
python3 -m export.onnx.ultralytics yolo26s-seg --save models/runtime/yolo26s-seg.onnx
```

### Python API

```python
from export.onnx import UltralyticsExporter

exporter = UltralyticsExporter("yolov8s-seg")
onnx_path = exporter.export_onnx("models/runtime/yolov8s-seg.onnx")
```

### 输入输出规格

| 项目 | 规范 |
|------|------|
| 输入名 | `image` |
| 输入 shape | `(1, 3, 640, 640)` |
| 输入 dtype | `float32` |
| 输入范围 | `[0, 1]`（LetterBox + ÷255 后） |
| 输出名 (检测) | `output0` |
| 输出 shape (检测) | `(1, 116, 8400)` |
| 输出名 (掩码) | `output1` |
| 输出 shape (掩码) | `(1, 32, 160, 160)` |

### 输出结构详解

```
# output0: 检测+掩码系数
pred[0, :, i] = [cx, cy, w, h, cls_0...cls_79, mask_0...mask_31]
                 ├── 4 bbox params
                 ├── 80 class scores (COCO)
                 └── 32 mask coefficients

# output1: 原型掩码 (Proto Mask)
shape: (1, 32, 160, 160)
├── 32 个原型掩码通道
└── 固定空间尺寸 160×160（与输入尺寸无关！）
```

### 掩码解码算法

```
1. sigmoid(tensordot(mask_coeffs[32], proto_masks[32, 160, 160])) → (160, 160)
2. Resize → 原始图像尺寸
3. Crop to bbox → 最终实例掩码
```

**关键点：** Proto mask 的 `160×160` 是固定的，不随输入尺寸变化。后处理时必须正确处理这个尺寸差异。

---

## L2: ONNX → TensorRT

### FP16 引擎

```bash
python3 -m export.tensorrt.build_fp16 \
    --onnx models/runtime/yolov8s-seg.onnx \
    --save models/tensorrt/yolov8s-seg_fp16.engine \
    --workspace 4
```

> **注意：** 分割模型的 ONNX 有 **两个输出**（`output0` + `output1`）。TensorRT 构建时自动处理多输出解析，无需额外配置。

### INT8 引擎（PyTorch 校准器）

```bash
# Step 1: 生成校准数据（COCO 子集）
python3 export/scripts/generate_calib_cache_for_coco.py \
    --input_dir ./cal_coco_src \
    --output_dir ./cal_coco_dst \
    --input_size 640 \
    --max_images 100

# Step 2: 构建 INT8 引擎
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/yolov8s-seg.onnx \
    --calib_dir ./cal_coco_dst \
    --output models/tensorrt/yolov8s-seg_int8.engine \
    --input_shape 1 3 640 640 \
    --workspace 4
```

### 校准数据生成

分割模型的校准数据使用与检测模型**完全相同的预处理管线**：

```bash
python3 export/scripts/generate_calib_cache_for_coco.py \
    --input_dir /path/to/coco_val2017 \
    --output_dir ./calib_dst \
    --input_size 640 \
    --max_images 100
```

**预处理管线：** `LetterBox(640) → BGR→RGB → /255 → CHW float32`

**输出文件大小：** `640 × 640 × 3 × 4 = ~4.7 MB`

---

## L3: Triton 配置

### config.pbtxt 生成

```bash
python3 -m export.triton.config_generator \
    --model-name Segment_COCO_YOLOv8sSeg_TRT \
    --backend tensorrt \
    --task segment \
    --save ./models/triton/
```

| 参数 | 说明 |
|------|------|
| `--task segment` | 自动配置分割输出 dims |

### 生成的 config.pbtxt

```protobuf
name: "Segment_COCO_YOLOv8sSeg_TRT"
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
    dims: [116, 8400]
  }
]
```

> **注意：** 当前 `TritonConfigGenerator` 为分割模型生成单输出配置（`output0`）。如果模型有两个输出（`output0` + `output1` proto mask），需手动在 `config.pbtxt` 中补充 `output1` 定义，或扩展 `_TASK_IO_SPECS` 中的 `segment` 条目。

### 手动补充 output1

```protobuf
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [116, 8400]
  },
  {
    name: "output1"
    data_type: TYPE_FP32
    dims: [32, 160, 160]
  }
]
```

---

## Triton 模型命名规范

```
<任务>_<数据集>_<模型架构>_<后端>

检测：Detect_COCO_YOLOv8s_TRT
分割：Segment_COCO_YOLOv8sSeg_TRT
```

| 字段 | 可选值 |
|------|--------|
| 任务 | Detect, Segment, Classify, Pose |
| 数据集 | COCO, ImageNet, Custom |
| 模型架构 | YOLOv8sSeg, YOLO11sSeg |
| 后端 | ONNX, TRT |

---

## 注意事项

1. **Proto mask 尺寸固定：** 无论输入尺寸多大，`output1` 始终是 `(1, 32, 160, 160)`。后处理的 mask resize 步骤必须正确处理。
2. **多输出验证：** 导出后验证时需同时对比 `output0` 和 `output1`，`compare_torch_onnx` 会逐个检查所有输出。
3. **精度损失：** 分割模型 INT8 量化的精度损失通常比检测模型略高（~1% mAP vs ~0.5–1% mAP），建议转换后进行全量 mAP 评估。
4. **FP16 掩码质量：** FP16 下掩码系数和 proto mask 的精度损失通常可忽略（< 0.1%），对最终掩码质量影响极小。

---

## 深入阅读

- [TensorRT 深入指南](tensorrt.md) — FP16/INT8 量化原理、双校准器对比、精度损失预期
- [Triton 部署指南](triton.md) — config.pbtxt 完整配置、自定义输出定义
- [`specs/export/onnx_export.md`](../specs/export/onnx_export.md) — Ultralytics 分割导出规范
