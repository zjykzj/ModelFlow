# Triton 部署指南

> config.pbtxt 配置详解、模型仓库管理、Docker 部署、dynamic batching。

## 模型仓库结构

Triton 要求严格的目录结构来组织模型：

```
models/triton/
└── <model_name>/                # 模型唯一标识
    ├── config.pbtxt             # 模型配置 (Protobuf Text)
    └── <version>/               # 版本号 (正整数)
        ├── model.onnx           # ONNX 后端
        └── model.plan           # 或 TensorRT 后端 (二选一)
```

**版本管理：** Triton 默认加载最大数字版本；可通过 `config.pbtxt` 中的 `version_policy` 精确控制。

---

## config.pbtxt 配置

### 配置生成工具

`TritonConfigGenerator`（`export/triton/config_generator.py`）根据任务类型自动生成输入输出 dims。

```python
from export.triton import TritonConfigGenerator

gen = TritonConfigGenerator(
    model_name="Detect_COCO_YOLOv8s_TRT",
    backend="tensorrt",          # "onnxruntime" | "tensorrt"
    task="detect",               # "classify" | "detect" | "segment" | "pose"
    max_batch_size=0,            # 0 = 无 batch 维度
    instance_count=1,            # 模型实例数（GPU 并发能力）
    dynamic_batching=False,      # 是否启用 dynamic batching
)
gen.save("models/triton/")
```

### 任务对应的输入输出规格

| 任务 | 输入 dims | 输出 dims | 说明 |
|------|-----------|-----------|------|
| `classify` | `[3, 224, 224]` | `[1000]` | ImageNet 分类 |
| `detect` | `[3, 640, 640]` | `[84, 8400]` | COCO 检测 |
| `segment` | `[3, 640, 640]` | `[116, 8400]` | COCO 分割 |
| `pose` | `[3, 640, 640]` | `[56, 8400]` | COCO 姿态 |

### 后端平台对应关系

| 用户输入 | config.pbtxt 中的 `platform` | 模型文件名 |
|---------|------------------------------|-----------|
| `onnxruntime` / `onnx` | `onnxruntime_onnx` | `model.onnx` |
| `tensorrt` / `trt` | `tensorrt_plan` | `model.plan` |

### 完整配置参数

| 参数 | 必填 | 默认 | 说明 |
|------|:---:|------|------|
| `model_name` | ✅ | — | 模型名（须与目录名一致） |
| `backend` | ✅ | `tensorrt` | `onnxruntime` / `tensorrt` |
| `task` | ✅ | `detect` | `classify` / `detect` / `segment` / `pose` |
| `max_batch_size` | — | `0` | 0 = 无 batch 维度 |
| `instance_count` | — | `1` | 每个 GPU 的模型实例数 |
| `dynamic_batching` | — | `False` | 启用请求动态合并 |
| `preferred_batch_size` | — | `[1, 4, 8]` | dynamic batch 偏好值 |
| `max_queue_delay_microseconds` | — | `100` | 最大排队延迟 (μs) |

---

## Dynamic Batching

动态 batching 将多个客户端请求自动合并为一个 batch，提升 GPU 利用率。

### 配置

```python
gen = TritonConfigGenerator(
    model_name="Detect_COCO_YOLOv8s_TRT",
    backend="tensorrt",
    task="detect",
    max_batch_size=8,            # ← 必须 > 0
    dynamic_batching=True,       # ← 启用
    preferred_batch_size=[1, 4, 8],
    max_queue_delay_microseconds=100,
)
gen.save("models/triton/")
```

### 生成的 config.pbtxt

```protobuf
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

> **前置条件：** TensorRT 引擎必须构建为支持 `max_batch_size` 的动态 shape。固定 shape 引擎无法使用 dynamic batching。

### 参数说明

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `max_batch_size` | 4–16 | 取决于 GPU 显存 |
| `preferred_batch_size` | `[1, 4, 8]` | Triton 优先按这些值合并 |
| `max_queue_delay` | 50–500 μs | 值越大可合并的请求越多，但延迟增加 |

---

## ModelRepoBuilder

`ModelRepoBuilder`（`export/triton/model_repo.py`）管理模型文件的部署和组织。

### 基础用法

```python
from export.triton import ModelRepoBuilder

builder = ModelRepoBuilder("models/triton/")

# 部署模型（自动根据后缀推断后端）
builder.deploy("Detect_COCO_YOLOv8s_TRT", "yolov8s_fp16.engine")
builder.deploy("Detect_COCO_YOLOv8s_ONNX", "yolov8s.onnx")

# 显式指定后端和版本
builder.deploy(
    model_name="Detect_COCO_YOLOv8s_TRT",
    model_file="models/tensorrt/yolov8s_fp16.engine",
    backend="tensorrt",
    version=1,
    overwrite=True,
)
```

### 模型命名规范

```python
# 手动命名
ModelRepoBuilder.build_model_name("Detect", "COCO", "YOLOv8s", "TRT")
# → "Detect_COCO_YOLOv8s_TRT"
```

| 字段 | 示例值 | 说明 |
|------|--------|------|
| 任务 | Detect, Classify, Segment, Pose | 首字母大写 |
| 数据集 | COCO, ImageNet, Custom | 首字母大写 |
| 模型架构 | YOLOv8s, EfficientNetB0 | 无下划线 |
| 后端 | ONNX, TRT | 大写 |

### 列出现有模型

```python
builder = ModelRepoBuilder("models/triton/")
models = builder.list_models()
# → 所有包含 config.pbtxt 的模型目录
```

---

## CLI 工作流

### 完整部署示例

```bash
# Step 1: 导出 ONNX
python3 -m export.onnx.ultralytics yolov8s --save models/runtime/yolov8s.onnx

# Step 2: 构建 TensorRT 引擎
python3 -m export.tensorrt.build_fp16 \
    --onnx models/runtime/yolov8s.onnx \
    --save models/tensorrt/yolov8s_fp16.engine

# Step 3: 生成 config.pbtxt
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt --task detect \
    --save ./models/triton/

# Step 4: 部署模型文件
python3 -c "
from export.triton import ModelRepoBuilder
ModelRepoBuilder('models/triton/').deploy(
    'Detect_COCO_YOLOv8s_TRT',
    'models/tensorrt/yolov8s_fp16.engine'
)
"

# Step 5: 启动 Triton
docker run --gpus=all -it --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models/triton:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
```

### 多模型 A/B 测试

```bash
# YOLOv8s FP16
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt --task detect --save ./models/triton/

# YOLOv8s INT8
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_INT8 \
    --backend tensorrt --task detect --save ./models/triton/

# YOLOv8s ONNX (基线)
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_ONNX \
    --backend onnxruntime --task detect --save ./models/triton/
```

---

## Docker 部署

### 启动命令

```bash
docker run --gpus=all -it --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models/triton:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
```

### 端口说明

| 端口 | 协议 | 用途 |
|------|------|------|
| 8000 | HTTP | RESTful API 推理 |
| 8001 | gRPC | gRPC 推理（高性能） |
| 8002 | HTTP | Prometheus 指标 |

### 模型状态检查

```bash
# 检查模型是否就绪
curl localhost:8000/v2/models/Detect_COCO_YOLOv8s_TRT/ready
# → 模型就绪时返回 200

# 查看模型元数据
curl localhost:8000/v2/models/Detect_COCO_YOLOv8s_TRT
```

### 推理请求

```python
import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient("localhost:8001")
# 后续使用 client.infer(...) 发送推理请求
```

---

## ONNX vs TensorRT 后端对比

| 对比项 | ONNX Runtime | TensorRT |
|--------|:-----------:|:--------:|
| 模型文件 | `model.onnx` | `model.plan` |
| 推理性能 | 基线 | 更优（FP16/INT8 加速） |
| 构建步骤 | 只需 ONNX 导出 | ONNX + TRT 构建（两次） |
| 灵活性 | 高 | 受引擎 shape 限制 |
| 加载速度 | 快 | 较慢（反序列化） |
| 适用场景 | 快速部署 / GPU 不支持 TRT | 生产环境 / 极致性能 |

---

## 验证清单

| 检查项 | 方法 |
|--------|------|
| 配置文件格式 | 启动 Triton，检查日志无语法错误 |
| 模型加载 | Triton 日志显示 `READY` 状态 |
| 推理验证 | 使用 Triton Python Client 对比与本地推理输出 |
| 模型就绪 | `curl localhost:8000/v2/models/<name>/ready` 返回 OK |

---

## 深入阅读

- [分类模型导出](classification.md) — 分类模型 Triton 部署示例
- [检测模型导出](detection.md) — 检测模型 Triton 部署示例
- [分割模型导出](segmentation.md) — 分割模型 Triton 部署 + 多输出配置
- [TensorRT 深入指南](tensorrt.md) — 量化策略与 Triton 后端性能优化
- [`specs/export/triton_deployment.md`](../specs/export/triton_deployment.md) — Triton 部署规范
