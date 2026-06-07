# TensorRT 转换规范

> **Status:** Draft
> **Version:** 0.1
> **前置阅读:** [`specs/export/onnx_export.md`](onnx_export.md)（ONNX 导出规范），TensorRT 基础概念

## 0. 版本要求

本模块基于 **TensorRT 10.x** API 实现，关键 API 差异：

| API | TensorRT 8.x | TensorRT 10.x |
|-----|-------------|---------------|
| 工作空间配置 | `config.max_workspace_size = N` | `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, N)` |
| 校准器接口 | `IInt8EntropyCalibrator2`（同） | `IInt8EntropyCalibrator2`（同） |
| 引擎构建 | `builder.build_serialized_network(network, config)` | 同 |

**安装：**
```bash
pip install tensorrt
# 或从 NVIDIA 官网下载对应 CUDA 版本的 .whl：
# https://developer.nvidia.com/tensorrt/download
```

> **注意：** TensorRT 10.x 构建的 `.engine` 文件无法在 TensorRT 8.x 环境中加载。部署时需确保 Triton Server 版本 ≥ 24.06（内置 TRT 10.x）。

## 1. TensorRT 工作原理

TensorRT 是 NVIDIA 的高性能深度学习推理优化器。它将训练好的模型（ONNX）转换为经过**图优化**和**内核调优**的推理引擎。

```
ONNX 计算图
    │
    ▼
┌──────────────────────────────┐
│      1. 图解析 (Graph Parse)   │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│   2. 图优化 (Graph Optimization) │
│   ├── 层融合 (Layer Fusion)     │
│   ├── 常量折叠 (Constant Folding)│
│   ├── 死分支消除                │
│   └── 精度校准 (INT8 Quantization)│
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  3. 内核自动调优 (Kernel Tuning) │
│   └── 为每层选择最优 CUDA kernel │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│    4. 序列化 → .engine 文件     │
└──────────────────────────────┘
```

**关键优化手段：**

| 优化 | 说明 | 效果 |
|------|------|------|
| **层融合** | 将 Conv+BN+ReLU 合并为一个 kernel | 减少显存带宽消耗和 kernel launch 开销 |
| **精度校准** | FP32 → INT8，通过 KL 散度找到最优量化阈值 | 2–3× 加速 |
| **内核调优** | 每种算子尝试多种 CUDA kernel，选择最优 | 最佳硬件利用 |
| **显存复用** | 分析张量生命周期，复用显存 | 降低显存峰值 |

## 2. FP16 转换

### 2.1 原理

FP16 (半精度浮点) 将 FP32 的 32 位权重和激活值截断为 16 位：

| 格式 | 符号位 | 指数位 | 尾数位 | 表示范围 |
|------|-------|-------|-------|---------|
| FP32 | 1 | 8 | 23 | ~3.4×10³⁸ |
| FP16 | 1 | 5 | 10 | ~6.6×10⁴ |

**精度损失：** 通常 ~0%（对于视觉模型几乎不可感知）

**加速比：** 1.5–2×（取决于模型和 GPU 架构）

### 2.2 构建方式

**方式 A：trtexec 命令行（推荐）**

```bash
trtexec --onnx=model.onnx \
        --saveEngine=model_fp16.engine \
        --workspace=4096 \
        --fp16
```

适合批量转换任务，输出可以直接用于推理和 Triton 部署。

**方式 B：Python API 封装**

对于需要集成到自动化管线中的场景，可以封装 trtexec 调用或使用 TensorRT Python API。

### 2.3 shape 配置

对于固定尺寸模型（如 640×640 检测），使用静态 shape：

```
--minShapes=input:1x3x640x640 \
--optShapes=input:1x3x640x640 \
--maxShapes=input:1x3x640x640
```

动态 shape 详见下文第 4 节。

### 2.4 适用场景

| 场景 | 推荐 |
|------|------|
| 有 Tesla T4/RTX 30/40 系列 GPU | ✅ 默认启用 |
| 精度要求极高（医疗等） | 先验证 FP16 与 FP32 差异 |
| 边缘设备 Jetson | ✅ 推荐 |
| GPU 不支持 FP16（较老架构） | ❌ 使用 FP32 |

## 3. INT8 转换

### 3.1 原理

INT8 量化将 FP32 值映射到 8 位整数范围 `[-128, 127]`：

```
FP32 value ──▶ 乘以量化因子 (scale) ──▶ round ──▶ INT8
INT8 value  ──▶ 乘以反量化因子 ──▶ FP32 (近似)
```

**校准 (Calibration)** 是 INT8 量化的核心：在小规模数据集上运行 FP32 推理，收集每层的激活值分布，使用 **KL 散度**找到最优的量化阈值，使量化前后的信息损失最小。

```
校准数据集 (50–100 张)
    │
    ▼
FP32 推理 ──▶ 收集激活直方图 ──▶ KL 散度优化 ──▶ 每层量化阈值
                                                      │
                                                      ▼
                                              INT8 引擎构建
```

### 3.2 校准数据集要求

| 模型类型 | 推荐数据集 | 校准图片数 | 预处理方式 |
|---------|-----------|-----------|-----------|
| 分类 (ImageNet) | ImageNet 验证集子集 | 50–100 | Resize(256) + CenterCrop(224) |
| 检测 (COCO) | COCO val2017 子集 | 50–100 | LetterBox(640) + /255 |
| 分割 | COCO val2017 子集 | 50–100 | 同检测预处理 |

**校准数据准备流程：**

```
原始图片 ──▶ 预处理 ──▶ 保存为 .bin (float32 裸二进制)
```

```bash
# 分类模型校准数据
python3 export/scripts/generate_calib_cache_for_imagenet.py \
    --input_dir export/cal_imagenet_src \
    --output_dir export/cal_imagenet_dst \
    --crop_size 224

# 检测模型校准数据
python3 export/scripts/generate_calib_cache_for_coco.py \
    --input_dir export/cal_coco_src \
    --output_dir export/cal_coco_dst \
    --input_size 640
```

### 3.3 双校准器策略

根据部署环境的依赖情况，提供两种校准器实现：

| 特性 | PyTorch Calibrator | PyCUDA Calibrator |
|------|:------------------:|:-----------------:|
| **依赖** | `torch` + `tensorrt` | `pycuda` + `tensorrt` |
| **环境启动** | 需加载 Torch 库 (稍慢) | 直接绑定 CUDA (快速) |
| **内存开销** | ~2GB 以上 | <100MB |
| **数据拷贝** | `torch.pin_memory` + `cuda.copy_` | `pycuda.pagelocked_empty` + `memcpy_htod` |
| **推荐设备** | RTX 服务器、开发机 | Jetson、嵌入式、Docker 精简镜像 |
| **调试便利性** | 高（Torch 生态） | 低（需 pycuda 知识） |

**数据自愈：** 两种校准器均实现以下保护机制：
- 自动检测并跳过大小不匹配的 `.bin` 文件
- 修复 NaN/Inf 数据 (`nan_to_num`)
- 逐文件异常捕获（单个文件损坏不影响整体构建）

### 3.4 构建方式

```bash
# PyTorch 环境（推荐开发机使用）
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir export/cal_coco_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640 \
    --workspace 4

# PyCUDA 环境（推荐 Jetson/嵌入式使用）
python3 -m export.tensorrt.build_int8_pycuda \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir export/cal_coco_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640
```

### 3.5 精度损失预期

| 模型类型 | FP32 → FP16 损失 | FP32 → INT8 损失 |
|---------|:----------------:|:----------------:|
| 分类 (ResNet) | < 0.1% | < 0.5% |
| 检测 (YOLOv8) | < 0.1% | ~0.5–1% mAP |
| 分割 | < 0.1% | < 1% mAP |

**注意：** 实际精度损失因模型和校准数据集而异，建议每次转换后进行精度验证。

## 4. 动态 Shape 处理

### 4.1 何时需要动态 Shape

| 场景 | 建议 |
|------|------|
| 推理时输入尺寸固定 (如 YOLOv8 640×640) | ✅ 使用静态 shape |
| 推理时输入尺寸可变 (如分类模型不同分辨率) | 使用动态 shape |
| 需要 batch 推理 | 使用动态 batch |

### 4.2 配置方式

```bash
# 动态 batch 示例
--minShapes=input:1x3x640x640 \
--optShapes=input:8x3x640x640 \
--maxShapes=input:16x3x640x640

# 动态宽高示例
--minShapes=input:1x3x224x224 \
--optShapes=input:1x3x640x640 \
--maxShapes=input:1x3x1280x1280
```

**注意：**
- 动态 shape 引擎的推理速度通常慢于固定 shape 引擎
- 动态 shape 的 INT8 校准更加复杂
- Triton 支持动态 shape 引擎

## 5. 量化策略选择

### 5.1 决策树

```
模型能否在 GPU 上运行？
├── 否 ──▶ ONNX Runtime (CPU) — 无需 TensorRT
└── 是 ──▶ 是否需要最大性能？
    ├── 否 ──▶ ONNX Runtime (GPU) — 或 FP16
    └── 是 ──▶ 精度容忍度如何？
        ├── 宽松 ──▶ INT8 (2–3× 加速)
        └── 严格 ──▶ FP16 (1.5–2× 加速)
```

### 5.2 对比速查

| 因素 | FP16 | INT8 |
|------|------|------|
| 加速收益 | 1.5–2× | 2–3× |
| 精度风险 | 极低 | 中低 |
| 校准数据 | 不需要 | 需要 50–100 张 |
| 构建时间 | 几分钟 | 几十分钟（校准耗时） |
| 适用场景 | 默认选择 | 极致性能、边缘设备 |
| 硬件要求 | FP16 支持 | INT8 支持 (Turing+) |

## 6. 验证

| 检查项 | 方法 | 标准 |
|--------|------|------|
| FP32 vs FP16 精度 | 同一输入对比输出 | 精度损失 < 0.1% |
| FP32 vs INT8 精度 | 同一输入对比输出 | 精度损失 < 1% (mAP) |
| 引擎加载 | `trtexec --loadEngine=model.engine` | 加载成功 |
| 推理稳定性 | 连续推理 100+ 次 | 输出稳定，无 NaN |
