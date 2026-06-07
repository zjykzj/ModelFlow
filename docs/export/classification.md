# 分类模型全链路导出

> 适用于 torchvision 分类模型：ResNet、MobileNet、EfficientNet、ViT 等。

## 支持的模型

| 系列 | 模型 | 默认输入 | 官方权重 |
|------|------|:-------:|:-------:|
| **ResNet** | resnet18, resnet34, resnet50, resnet101, resnet152 | 224×224 | ImageNet-1K |
| **EfficientNet** | efficientnet_b0 ~ b7 | 224~600 | ImageNet-1K |
| **EfficientNetV2** | efficientnet_v2_s / m / l | 224~480 | ImageNet-1K |
| **MobileNet** | mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small | 224 | ImageNet-1K |
| **ShuffleNet** | shufflenet_v2_x0_5 / x1_0 / x1_5 / x2_0 | 224 | ImageNet-1K |
| **SqueezeNet** | squeezenet1_0, squeezenet1_1 | 224 | ImageNet-1K |
| **MNASNet** | mnasnet0_5 / 0_75 / 1_0 / 1_3 | 224 | ImageNet-1K |
| **DenseNet** | densenet121 / 161 / 169 / 201 | 224 | ImageNet-1K |
| **VGG** | vgg11 / 13 / 16 / 19 | 224 | ImageNet-1K |
| **ConvNeXt** | convnext_tiny / small / base / large | 224 | ImageNet-1K |
| **ViT** | vit_b_16, vit_b_32, vit_l_16, vit_l_32 | 224 | ImageNet-1K |

## 全链路总览

```
torchvision 预训练模型 (PT)
    │
    ├── L1 ──▶ export.onnx.convert ──▶ .onnx ──▶ ONNX Runtime 推理
    │
    ├── L2 ──▶ export.tensorrt.build_fp16 ──▶ .engine (FP16)
    │   └──▶ export.tensorrt.build_int8 ──▶ .engine (INT8) ← 需校准数据
    │
    └── L3 ──▶ export.triton.config_generator ──▶ Triton 部署
```

---

## L1: PT → ONNX

### 导出器：`TorchvisionExporter`

位于 `export/onnx/convert.py`，继承自 `BaseExporter`。

**导出流程：**

1. 加载 torchvision 预训练模型（自动适配 torchvision < 0.13 的 `pretrained=True` 和 ≥ 0.13 的 `weights=` 参数）
2. 构建 dummy input `(1, C, H, W)`
3. 运行一次前向，确定输出结构
4. `torch.onnx.export` 导出，启用常量折叠
5. 自动验证：`onnx.checker.check_model` + PT vs ONNX 输出对比 (`rtol=1e-3, atol=1e-5`)

### CLI 参数

```
python3 -m export.onnx.convert --help
```

| 参数 | 必填 | 默认 | 说明 |
|------|:---:|------|------|
| `--model` | ✅ | — | torchvision 模型名（如 `efficientnet_b0`） |
| `--save` | ✅ | — | ONNX 保存路径 |
| `--img-size` | — | `[224]` | 输入尺寸，单个值或 `H W` |
| `--img-channels` | — | `3` | 输入通道数 |
| `--opset` | — | `12` | ONNX opset 版本 |

### 使用示例

```bash
# ResNet
python3 -m export.onnx.convert \
    --model resnet50 \
    --save models/runtime/resnet50.onnx \
    --img-size 224

# EfficientNet
python3 -m export.onnx.convert \
    --model efficientnet_b0 \
    --save models/runtime/efficientnet_b0.onnx \
    --img-size 224

# MobileNet V2
python3 -m export.onnx.convert \
    --model mobilenet_v2 \
    --save models/runtime/mobilenet_v2.onnx \
    --img-size 224

# MobileNet V3-Large（非正方形输入）
python3 -m export.onnx.convert \
    --model mobilenet_v3_large \
    --save models/runtime/mobilenet_v3_large.onnx \
    --img-size 224 224
```

### Python API

```python
from export.onnx import TorchvisionExporter

# 基础用法
exporter = TorchvisionExporter("resnet50", opset=12)
onnx_path = exporter.export_onnx(
    output_path="models/runtime/resnet50.onnx",
    img_size=224,
)

# 自定义尺寸和验证
exporter = TorchvisionExporter("efficientnet_b0")
onnx_path = exporter.export_onnx(
    output_path="models/runtime/efficientnet_b0.onnx",
    img_size=(224, 224),
    do_validation=True,  # 默认开启，验证 ONNX 格式合法 + 输出数值一致
)
```

### 输入输出规格

| 项目 | 规范 |
|------|------|
| 输入名 | `image` |
| 输入 shape | `(1, 3, 224, 224)` |
| 输入 dtype | `float32` |
| 输入范围 | Normalize 后，均值 0 / 方差 1 |
| 输出名 | `output0` |
| 输出 shape | `(1, 1000)` |
| 输出含义 | logits（未经过 softmax） |

---

## L2: ONNX → TensorRT

### FP16 引擎

```bash
# trtexec 优先，Python API 兜底
python3 -m export.tensorrt.build_fp16 \
    --onnx models/runtime/resnet50.onnx \
    --save models/tensorrt/resnet50_fp16.engine \
    --workspace 4
```

| 参数 | 必填 | 默认 | 说明 |
|------|:---:|------|------|
| `--onnx` | ✅ | — | ONNX 模型路径 |
| `--save` | ✅ | — | 引擎保存路径 |
| `--workspace` | — | `4` | GPU 工作空间 (GB) |
| `--no-trtexec` | — | `false` | 跳过 trtexec，直接用 Python API |
| `--shapes` | — | — | 动态 shape（如 `input:1x3x224x224`） |

### INT8 引擎（PyTorch 校准器）

```bash
# Step 1: 生成校准数据
python3 export/scripts/generate_calib_cache_for_imagenet.py \
    --input_dir ./cal_imagenet_src \
    --output_dir ./cal_imagenet_dst \
    --crop_size 224 \
    --max_images 100

# Step 2: 构建 INT8 引擎
python3 -m export.tensorrt.build_int8 \
    --onnx models/runtime/resnet50.onnx \
    --calib_dir ./cal_imagenet_dst \
    --output models/tensorrt/resnet50_int8.engine \
    --input_shape 1 3 224 224 \
    --workspace 4
```

| 参数 | 必填 | 默认 | 说明 |
|------|:---:|------|------|
| `--onnx` | ✅ | — | ONNX 模型路径 |
| `--calib_dir` | ✅ | — | 校准 `.bin` 数据目录 |
| `--output` | ✅ | — | 引擎保存路径 |
| `--input_shape` | — | `1 3 640 640` | 输入 shape N C H W |
| `--workspace` | — | `4` | GPU 工作空间 (GB) |
| `--cache_file` | — | `calib_torch.cache` | 校准缓存文件名 |

### INT8 引擎（PyCUDA 校准器，Jetson/嵌入式）

```bash
python3 -m export.tensorrt.build_int8_pycuda \
    --onnx models/runtime/resnet50.onnx \
    --calib_dir ./cal_imagenet_dst \
    --output models/tensorrt/resnet50_int8.engine \
    --input_shape 1 3 224 224
```

### 校准数据生成

```bash
python3 export/scripts/generate_calib_cache_for_imagenet.py \
    --input_dir /path/to/imagenet_val \
    --output_dir ./calib_dst \
    --crop_size 224 \
    --resize_size 256 \
    --max_images 100 \
    --format bin
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input_dir` | (必填) | 源图片目录 |
| `--output_dir` | (必填) | 输出 `.bin` 目录 |
| `--crop_size` | `224` | 裁剪尺寸 |
| `--resize_size` | `256` | 缩放尺寸 |
| `--format` | `bin` | 输出格式 (`bin` / `npy`) |
| `--max_images` | `100` | 最大处理数（50-100 即可） |

**预处理管线：** `BGR→RGB → Resize(256) → CenterCrop(224) → Normalize(mean,std) → CHW float32`

**输出文件大小：** `224 × 224 × 3 × 4 = 602,112 Bytes`

---

## L3: Triton 配置

### config.pbtxt 生成

```bash
python3 -m export.triton.config_generator \
    --model-name Classify_ImageNet_ResNet50_TRT \
    --backend tensorrt \
    --task classify \
    --save ./models/triton/
```

| 参数 | 必填 | 默认 | 说明 |
|------|:---:|------|------|
| `--model-name` | ✅ | — | 模型名称（`Classify_ImageNet_ResNet50_TRT`） |
| `--backend` | ✅ | `tensorrt` | `onnxruntime` / `tensorrt` |
| `--task` | ✅ | `detect` | `classify` / `detect` / `segment` / `pose` |
| `--save` | ✅ | — | 模型仓库根目录 |
| `--max-batch` | — | `0` | 最大 batch（0 = 无 batch 维度） |
| `--dynamic-batch` | — | — | 启用 dynamic batching |
| `--instance-count` | — | `1` | 模型实例数 |

### 部署模型文件

```bash
# 使用 ModelRepoBuilder（或手动复制）
python3 -c "
from export.triton import ModelRepoBuilder
builder = ModelRepoBuilder('models/triton/')
builder.deploy('Classify_ImageNet_ResNet50_TRT', 'resnet50_fp16.engine')
"
```

### 生成的目录结构

```
models/triton/
└── Classify_ImageNet_ResNet50_TRT/
    ├── config.pbtxt        # 自动生成
    └── 1/
        └── model.plan      # FP16/INT8 引擎文件
```

### 生成的 config.pbtxt 内容

```protobuf
name: "Classify_ImageNet_ResNet50_TRT"
platform: "tensorrt_plan"
max_batch_size: 0

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [1000]
  }
]
```

---

## ONNX 优化（可选）

导出后可以简化 ONNX 计算图：

```python
from export.onnx import optimize_onnx
optimize_onnx("model.onnx", "model_simplified.onnx")
```

简化内容：移除冗余 Identity 节点、合并相邻 Reshape/Transpose、消除无用 Cast 操作。

---

## 版本兼容性

| torchvision 版本 | 权重加载 | 处理方式 |
|:---:|------|------|
| < 0.13 | `model(pretrained=True)` | legacy 参数 |
| ≥ 0.13 | `model(weights=WeightsEnum)` | 自动查 `_TV_WEIGHTS_MAPPING` 映射表 |

---

## 深入阅读

- [TensorRT 深入指南](tensorrt.md) — FP16/INT8 量化原理、双校准器对比、性能调优
- [Triton 部署指南](triton.md) — config.pbtxt 完整配置、dynamic batching、Docker 部署
- [`specs/export/onnx_export.md`](../specs/export/onnx_export.md) — ONNX 导出原理与规范
