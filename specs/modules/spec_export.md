# 模型导出模块规格

> **Version:** 0.1
> **Status:** Draft
> **Dependencies:** `spec_architecture.md`（独立模块设计原则）

## 1. 模块定位

`export/` 是一个**独立模块**，不依赖 `modelflow/` 或 `cpp/` 的任何代码。职责是将 PyTorch 模型转换为 ONNX，再进一步转换为 TensorRT 引擎，以及生成 Triton 模型仓库配置。

```
PyTorch (.pt) ──▶ ONNX (.onnx) ──▶ TensorRT (.engine) FP16 / INT8
                                      │
                                      ▼
                               Triton 模型仓库 (config.pbtxt + model)
```

## 2. 目录结构

```
export/
├── core/                           # 导出基础设施
│   ├── base.py                     # BaseExporter（可选统一接口）
│   ├── validation.py               # ONNX checker + 输出对比
│   └── utils.py                    # 工具函数
├── onnx/                           # PyTorch → ONNX
│   ├── convert.py                  # 通用导出（torchvision models）
│   ├── ultralytics.py              # Ultralytics YOLO 导出
│   └── optimize.py                 # ONNX 优化（simplifier）
├── tensorrt/                       # ONNX → TensorRT
│   ├── build_fp16.py               # trtexec --fp16 封装
│   ├── build_int8.py               # INT8 构建器
│   └── calibrator.py               # INT8 校准器
├── triton/                         # Triton 配置生成
│   ├── config_generator.py         # config.pbtxt 生成
│   └── model_repo.py              # 仓库目录结构生成
├── scripts/                        # 辅助脚本
│   ├── generate_calib_cache_for_coco.py
│   ├── generate_calib_cache_for_imagenet.py
│   └── random_copy_images.py
└── tests/
    ├── test_export.py
    └── test_engine.py
```

## 3. 导出管线

### 3.1 PyTorch → ONNX

```bash
# 通用 torchvision 模型
python3 -m export.onnx.convert \
    --model efficientnet_b0 \
    --save models/runtime/efficientnet_b0.onnx \
    --img-size 224

# YOLO 模型
python3 -m export.onnx.ultralytics \
    --weights yolov8s.pt \
    --save models/runtime/yolov8s.onnx \
    --opset 12
```

**关键参数**：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--opset` | ONNX opset 版本 | 12 |
| `--img-size` | 输入图像尺寸 | 模型相关 |
| `--dynamic-batch` | 是否启用动态 batch | False |

**验证**：
- ONNX checker：`onnx.checker.check_model()`
- 输出对比：PyTorch vs ONNX Runtime，`rtol=1e-3, atol=1e-5`

### 3.2 ONNX → TensorRT FP16

```bash
python3 -m export.tensorrt.build_fp16 \
    --onnx models/runtime/yolov8s.onnx \
    --save models/tensorrt/yolov8s_fp16.engine \
    --workspace 4
```

**实现方式**：封装 `trtexec` 或 TensorRT Python API

```
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --workspace=4096 \
        --fp16 \
        --minShapes=input:1x3x640x640 \
        --optShapes=input:1x3x640x640 \
        --maxShapes=input:1x3x640x640
```

### 3.3 ONNX → TensorRT INT8

```bash
# 1. 生成校准缓存
python3 export/scripts/generate_calib_cache_for_coco.py

# 2. 构建 INT8 引擎
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir export/cal_coco_dst \
    --save models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640
```

**INT8 校准器**：

| 实现 | 依赖 | 适用场景 |
|------|------|----------|
| PyTorch Calibrator | `torch` | RTX 服务器、有 PyTorch 环境 |
| PyCUDA Calibrator | `pycuda` | Jetson、嵌入式、轻量环境 |

### 3.4 Triton 配置

```bash
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s \
    --backend tensorrt \              # 或 onnxruntime
    --task detect \
    --save models/triton/

# 生成结构:
# models/triton/Detect_COCO_YOLOv8s/
# ├── config.pbtxt
# └── 1/
#     └── model.engine (或 model.onnx)
```

## 4. 任务导出规范

| 任务 | 输入 shape | 输出说明 |
|------|-----------|----------|
| Classification | `(1, 3, 224, 224)` | logits `(1, num_classes)` |
| Detection | `(1, 3, 640, 640)` | pred `(1, num_dets, 5+nc)` |
| Instance Segmentation | `(1, 3, 640, 640)` | pred `(1, num_dets, 5+nc)` + proto `(1, 32, 160, 160)` |
| Semantic Segmentation | `(1, 3, H, W)` | logits `(1, num_classes, H, W)` |
| Multi-modal (CLIP) | `(1, 3, 224, 224)` + `text_ids` | image_embed `(1, 512)` + text_embed `(N, 512)` |

## 5. 精度验证

| 检查项 | 方法 | 标准 |
|--------|------|------|
| ONNX 合法性 | `onnx.checker.check_model()` | 不报错 |
| PT vs ONNX | 同一输入对比输出 | rtol=1e-3, atol=1e-5 |
| FP32 vs FP16 | 同一输入对比输出 | FP16 精度损失 < 0.1% |
| FP32 vs INT8 | 同一输入对比输出 | INT8 精度损失 < 1% |
