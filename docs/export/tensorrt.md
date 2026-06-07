# TensorRT 深入指南

> FP16 / INT8 量化原理、双校准器选型、性能调优。

## 工作原理

TensorRT 将 ONNX 计算图转换为针对特定 GPU 架构优化的推理引擎：

```
ONNX 计算图
    │
    ▼
┌──────────────────────────────┐
│  1. 图解析 (Graph Parse)      │  ← ONNX Parser 解析算子
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  2. 图优化 (Graph Opt)        │  ← 层融合 / 常量折叠 / 死分支消除
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  3. 内核调优 (Kernel Tuning)   │  ← 为每层选择最优 CUDA kernel
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  4. 序列化 → .engine 文件      │
└──────────────────────────────┘
```

| 优化手段 | 效果 | 说明 |
|---------|------|------|
| **层融合** | 减少显存带宽 + kernel launch 开销 | Conv+BN+ReLU → 单个 kernel |
| **精度校准** | 2–3× 加速 (INT8) | KL 散度找最优量化阈值 |
| **内核调优** | 最佳硬件利用 | 每种算子试多种 CUDA kernel |
| **显存复用** | 降低显存峰值 | 分析张量生命周期，复用 buffer |

---

## FP16 转换

### 原理

FP16 将 32 位权重和激活值截断为 16 位：

| 格式 | 符号位 | 指数位 | 尾数位 | 表示范围 |
|------|:-----:|:-----:|:-----:|------|
| FP32 | 1 | 8 | 23 | ~3.4×10³⁸ |
| FP16 | 1 | 5 | 10 | ~6.6×10⁴ |

- **精度损失：** 对于视觉模型通常 < 0.1%，几乎不可感知
- **加速比：** 1.5–2×（取决于模型和 GPU 架构）
- **零准备成本：** 无需校准数据，开箱即用

### 两种构建方式

| 方式 | 实现 | 适用场景 |
|------|------|---------|
| **trtexec** (推荐) | `subprocess` 调用 `trtexec` CLI | 批量转换、脚本自动化 |
| **Python API** | 直接使用 `tensorrt.Builder` + `OnnxParser` | 集成到 Python 管线、trtexec 不可用时 |

构建逻辑（`export/tensorrt/build_fp16.py`）：

```python
def build_fp16_engine(onnx_path, output_path, workspace=4, use_trtexec=True, shapes=None):
    if use_trtexec:
        if _build_via_trtexec(onnx_path, output_path, workspace, shapes):
            return True
        # trtexec 失败自动回退到 Python API
    return _build_via_python_api(onnx_path, output_path, workspace)
```

### CLI

```bash
python3 -m export.tensorrt.build_fp16 \
    --onnx model.onnx \
    --save model_fp16.engine \
    --workspace 4
```

| 参数 | 必填 | 默认 | 说明 |
|------|:---:|------|------|
| `--onnx` | ✅ | — | ONNX 模型路径 |
| `--save` | ✅ | — | 引擎保存路径 |
| `--workspace` | — | `4` | GPU 工作空间 (GB) |
| `--no-trtexec` | — | `false` | 跳过 trtexec，直接 Python API |
| `--shapes` | — | — | 动态 shape 字符串 |

### Python API

```python
from export.tensorrt import build_fp16_engine

build_fp16_engine(
    onnx_path="model.onnx",
    output_path="model_fp16.engine",
    workspace=4,
    use_trtexec=False,    # 强制使用 Python API
)
```

### Shape 配置

```bash
# 静态 shape（推荐，固定尺寸模型无需配置）

# 显式静态 shape
python3 -m export.tensorrt.build_fp16 \
    --onnx model.onnx --save model.engine \
    --shapes "input:1x3x640x640"
```

---

## INT8 转换

### 量化原理

```
FP32 → 收集激活值分布 (校准) → KL 散度优化 → 每层量化阈值 → INT8 引擎

INT8 推理时:
  INT8 weight × INT8 activation → INT32 累加 → 反量化 → FP32 输出
```

### 双校准器对比

| 特性 | **TorchCalibrator** | **PyCudaCalibrator** |
|------|:------------------:|:-------------------:|
| 依赖 | `torch` + `tensorrt` | `pycuda` + `tensorrt` |
| 内存开销 | ~2GB+ | < 100MB |
| 启动速度 | 较慢（加载 Torch 库） | 快（直绑 CUDA 驱动） |
| 数据拷贝 | `pin_memory` + `cuda.copy_` | `pagelocked_empty` + `memcpy_htod` |
| 适用环境 | RTX 服务器 / AutoDL / 开发机 | Jetson / 嵌入式 / Docker 精简镜像 |
| 调试便利性 | 高（Torch 生态） | 低（需 pycuda 知识） |

两者继承同一基类 `BaseCalibrator(trt.IInt8EntropyCalibrator2)`，共享：

| 能力 | 说明 |
|------|------|
| 文件扫描 | `.bin` 递归收集，按文件名排序 |
| 数据自愈 | NaN/Inf → `nan_to_num`；尺寸不匹配自动跳过 |
| 进度日志 | 每 10% 进度输出 |
| 校准缓存 | 写入磁盘缓存，下次构建可复用 |

### 构建 CLI

```bash
# PyTorch 校准器（开发机推荐）
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir ./calib_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640 \
    --workspace 4

# PyCUDA 校准器（Jetson 推荐）
python3 -m export.tensorrt.build_int8_pycuda \
    --onnx models/runtime/yolov8s.onnx \
    --calib_dir ./calib_dst \
    --output models/tensorrt/yolov8s_int8.engine \
    --input_shape 1 3 640 640
```

| 参数 | 必填 | 默认 | 说明 |
|------|:---:|------|------|
| `--onnx` | ✅ | — | ONNX 路径 |
| `--calib_dir` | ✅ | — | 校准 `.bin` 目录 |
| `--output` | ✅ | — | 引擎保存路径 |
| `--input_shape` | — | `1 3 640 640` | N C H W |
| `--workspace` | — | `4` | GPU 工作空间 GB |
| `--cache_file` | — | `calib_torch.cache` / `calib_pycuda.cache` | 缓存文件名 |

### Python API

```python
from export.tensorrt import build_int8_engine_torch

build_int8_engine_torch(
    onnx_path="model.onnx",
    calib_dir="./calib_dst",
    output_path="model_int8.engine",
    input_shape=(1, 3, 640, 640),
    workspace=4,
    cache_file="calib_torch.cache",
)
```

---

## 校准数据

### 数据要求

| 模型类型 | 推荐数据集 | 图片数 | 生成脚本 | 每图大小 (.bin) |
|---------|-----------|:-----:|---------|:----:|
| 分类 (224²) | ImageNet 验证集子集 | 50–100 | `generate_calib_cache_for_imagenet.py` | ~588 KB |
| 检测/分割 (640²) | COCO val2017 子集 | 50–100 | `generate_calib_cache_for_coco.py` | ~4.7 MB |

### 分类校准数据生成

```bash
python3 export/scripts/generate_calib_cache_for_imagenet.py \
    --input_dir ./cal_src \
    --output_dir ./cal_dst \
    --crop_size 224 \
    --resize_size 256 \
    --max_images 100 \
    --format bin
```

预处理管线：`BGR→RGB → Resize(256) → CenterCrop(224) → Normalize(mean,std) → CHW float32`

### 检测/分割校准数据生成

```bash
python3 export/scripts/generate_calib_cache_for_coco.py \
    --input_dir ./cal_src \
    --output_dir ./cal_dst \
    --input_size 640 \
    --max_images 100 \
    --format bin
```

预处理管线：`LetterBox(640) → BGR→RGB → /255 → CHW float32`

---

## 精度损失预期

| 模型类型 | FP32 → FP16 | FP32 → INT8 |
|---------|:----------:|:----------:|
| 分类 (ResNet/EfficientNet) | < 0.1% top-1 | < 0.5% top-1 |
| 检测 (YOLOv8/YOLO11) | < 0.1% mAP | ~0.5–1% mAP |
| 分割 (YOLOv8-seg) | < 0.1% mAP | < 1% mAP |

> 实际精度损失因模型和校准数据集而异，建议每次转换后进行全量精度验证。

---

## 动态 Shape

### 何时使用

| 场景 | 建议 |
|------|------|
| 推理时输入尺寸固定（YOLO 640²） | ✅ 静态 shape — 速度更快 |
| 推理时输入尺寸可变（分类多分辨率） | 动态 shape |
| 需要 batch 推理 | 动态 batch |

### 配置示例

```bash
# 动态 batch
--minShapes=input:1x3x640x640 \
--optShapes=input:8x3x640x640 \
--maxShapes=input:16x3x640x640

# 动态宽高
--minShapes=input:1x3x224x224 \
--optShapes=input:1x3x640x640 \
--maxShapes=input:1x3x1280x1280
```

**代价：** 动态 shape 引擎推理速度通常比固定 shape 引擎慢，且 INT8 校准更复杂。

---

## 验证

| 检查项 | 方法 | 标准 |
|--------|------|------|
| FP32 vs FP16 精度 | 同一输入对比输出 | < 0.1% 差异 |
| FP32 vs INT8 精度 | 同一输入对比输出 | < 1% mAP 损失 |
| 引擎加载 | `trtexec --loadEngine=model.engine` | 加载成功 |
| 推理稳定性 | 连续推理 100+ 次 | 输出稳定，无 NaN |

```bash
# 快速加载测试
trtexec --loadEngine=models/tensorrt/yolov8s_fp16.engine --iterations=100
```

---

## 量化策略决策

```
是否需要 GPU 推理？
├── 否 → ONNX Runtime (CPU) — 无需 TensorRT
└── 是 → 是否需要极致性能？
    ├── 否 → ONNX Runtime (GPU) 或 FP16
    └── 是 → 精度容忍度？
        ├── 严格 (< 0.5%) → FP16
        └── 宽松 (0.5–1%) → INT8
```

---

## 深入阅读

- [分类模型导出](classification.md) — 分类模型 FP16/INT8 构建示例
- [检测模型导出](detection.md) — 检测模型 FP16/INT8 构建示例
- [分割模型导出](segmentation.md) — 分割模型 FP16/INT8 构建示例
- [Triton 部署指南](triton.md) — TensorRT 引擎的 Triton 部署
- [`specs/export/tensorrt_conversion.md`](../specs/export/tensorrt_conversion.md) — TensorRT 转换规范
