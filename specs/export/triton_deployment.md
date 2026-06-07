# Triton 部署配置规范

> **Status:** Draft
> **Version:** 0.1
> **前置阅读:** [`specs/export/onnx_export.md`](onnx_export.md)，[`specs/export/tensorrt_conversion.md`](tensorrt_conversion.md)

## 1. Triton 模型仓库

Triton Inference Server 使用特定的目录结构来组织模型，称为**模型仓库 (Model Repository)**。

### 1.1 标准结构

```
models/triton/
└── <model_name>/                    # 模型名称（唯一标识）
    ├── config.pbtxt                 # 模型配置（文本格式）
    └── <version>/                   # 版本号（正整数）
        ├── model.onnx               # ONNX 模型文件
        └── model.plan               # 或 TensorRT 引擎文件（二选一）
```

**版本管理：**
- 每个模型可以有多个版本（1/, 2/, 3/, ...）
- Triton 默认加载最新版本（最大数字）
- 通过 `config.pbtxt` 中的 `version_policy` 控制

### 1.2 模型命名规范

```
<任务>_<数据集>_<模型架构>_<后端>

例如：
- Detect_COCO_YOLOv8s_ONNX
- Classify_ImageNet_EfficientNetB0_TRT
- Segment_COCO_YOLOv8sSeg_ONNX
```

| 字段 | 可选值 | 说明 |
|------|-------|------|
| 任务 | Detect, Classify, Segment, Pose, SemanticSeg | 任务类型 |
| 数据集 | COCO, ImageNet, Custom | 训练数据集 |
| 模型架构 | YOLOv8s, EfficientNetB0, ResNet50 | 模型名称 |
| 后端 | ONNX, TRT | 推理后端 |

## 2. config.pbtxt 配置

### 2.1 ONNX 后端配置

```
name: "Detect_COCO_YOLOv8s_ONNX"
platform: "onnxruntime_onnx"
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

### 2.2 TensorRT 后端配置

```
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

### 2.3 动态 batch 配置

需要启动 dynamic batching 时：

```
name: "Detect_COCO_YOLOv8s_TRT"
platform: "tensorrt_plan"
max_batch_size: 8

dynamic_batching {
  preferred_batch_size: [1, 4, 8]
  max_queue_delay_microseconds: 100
}

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

**注意：** 启用 dynamic batching 要求 ONNX/TensorRT 引擎支持 `max_batch_size`。TensorRT 引擎构建时需配置对应的动态 shape 参数。

### 2.4 参数说明

| 配置项 | 说明 | 必填 |
|--------|------|:----:|
| `name` | 模型名称，须与目录名一致 | ✅ |
| `platform` | 后端类型：`onnxruntime_onnx` 或 `tensorrt_plan` | ✅ |
| `max_batch_size` | 最大 batch（0=无batch维度，即NCHW中的N已在引擎内固定） | ✅ |
| `input` | 输入定义（name, data_type, dims） | ✅ |
| `output` | 输出定义 | ✅ |
| `dynamic_batching` | 动态 batch 策略 | 可选 |
| `instance_group` | 多实例配置 | 可选 |
| `version_policy` | 版本管理策略 | 可选 |

## 3. ONNX vs TensorRT 后端对比

| 对比项 | ONNX Runtime 后端 | TensorRT 后端 |
|--------|:----------------:|:-------------:|
| 模型文件 | `model.onnx` | `model.plan` |
| 推理性能 | 基线 | 更优（FP16/INT8） |
| 构建步骤 | 只需 ONNX 导出 | 需先 ONNX 导出 + TensorRT 构建 |
| 灵活性 | 高 | 受引擎 shape 限制 |
| 加载速度 | 快 | 相对较慢（需反序列化） |
| 适用场景 | 快速部署、GPU 较新支持 FP16 | 极致优化、生产环境 |

## 4. 启动 Triton Server

```bash
docker run --gpus=all -it \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models/triton:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
```

**端口说明：**

| 端口 | 协议 | 用途 |
|------|------|------|
| 8000 | HTTP | RESTful API 推理 |
| 8001 | gRPC | gRPC 推理（高性能） |
| 8002 | HTTP | Prometheus 指标 |

## 5. 配置文件自动生成

模型仓库结构和 `config.pbtxt` 应由脚本自动生成，避免手动编写导致的错误。

```bash
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt \
    --task detect \
    --save models/triton/

# 生成结构：
# models/triton/Detect_COCO_YOLOv8s_TRT/
# ├── config.pbtxt
# └── 1/
#     └── model.plan
```

### 5.1 生成规则

| 参数 | 说明 | 影响 |
|------|------|------|
| `--model-name` | 模型名称 | 目录名 + config.name |
| `--backend` | onnxruntime / tensorrt | platform + 模型文件扩展名 |
| `--task` | detect / classify / segment | 输入输出 dims |
| `--version` | 版本号 | 子目录名 |
| `--max-batch` | 最大 batch | max_batch_size + dynamic_batching |
| `--save` | 输出路径 | 仓库根目录 |

## 6. 验证

| 检查项 | 方法 |
|--------|------|
| 配置文件格式 | `tritonserver --model-repository=/models --strict-model-config=false` 启动日志 |
| 模型加载 | Triton 日志中应显示 `"READY"` 状态 |
| 推理验证 | 使用 Triton Python Client 发送请求对比输出 |
