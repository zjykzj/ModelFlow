# ONNX 导出规范

> **Status:** Draft
> **Version:** 0.1
> **前置阅读:** `specs/modules/spec_export.md`（导出模块架构），`specs/export/index.md`（导出知识库总览）

## 1. 概述

ONNX (Open Neural Network Exchange) 是 PyTorch 模型导出流程的**第一站**，也是后续 TensorRT 转换和 Triton 部署的前置条件。ONNX 作为模型的中间表示（IR），具有：

| 特性 | 说明 |
|------|------|
| **框架中立** | 不绑定 PyTorch，C++/Python/JS 均可加载 |
| **算子标准化** | 通过 ONNX opset 定义标准算子集 |
| **可优化** | 支持常量折叠、节点融合等图优化 |
| **可量化** | INT8/FP16 量化均基于 ONNX 计算图 |

## 2. 导出机制

### 2.1 `torch.onnx.export` 的工作原理

PyTorch 导出 ONNX 采用 **Tracing** 机制：

```
输入样例张量 ──▶ 模型前向传播 ──▶ 记录实际执行的算子 ──▶ ONNX 计算图
```

```python
torch.onnx.export(
    model=model,              # PyTorch 模型
    args=dummy_input,         # 样例输入（决定输入 shape 和 trace 路径）
    f=model.onnx,             # 输出路径
    input_names=["image"],    # 输入节点命名
    output_names=["output0"], # 输出节点命名
    opset_version=12,         # ONNX 算子集版本
    dynamic_axes=None,        # 动态轴设置
    do_constant_folding=True  # 常量折叠优化
)
```

**Tracing 的限制：**
- 只能记录**实际走到的**分支——如果模型中有 `if` 语句且条件依赖输入，另一条分支不会出现在 ONNX 中
- 控制流（循环、条件）会被展开为静态图
- 对动态 shape 模型（如不同尺寸输入）需配合 `dynamic_axes` 参数

### 2.2 opset 版本

ONNX opset 定义了可用的算子集合，版本越高支持的算子越多：

| opset | PyTorch 版本 | 说明 |
|-------|-------------|------|
| 11 | 1.5+ | 广泛支持，兼容性好 |
| **12** | 1.6+ | ✅ **推荐默认版本**，平衡算子支持与兼容性 |
| 13 | 1.8+ | 新增软硬件算子 |
| 14 | 1.10+ | 新增算子 |
| 15 | 1.11+ | 更多量化算子支持 |
| 16+ | 1.12+ | 持续演进 |

**选择原则：**
- **默认 opset=12**，这是当前最稳健的选择
- TensorRT 7.X 对 opset 12 支持最佳
- 如果遇到不支持的算子，可尝试升级 opset 版本
- 更高的 opset 不一定更好——可能降低目标后端的兼容性

## 3. torchvision 分类模型导出

### 3.1 支持的模型列表

| 系列 | 模型 | 默认输入尺寸 | 说明 |
|------|------|-------------|------|
| **ResNet** | resnet18, resnet34, resnet50, resnet101, resnet152 | 224×224 | 经典分类骨干 |
| **ResNeXt** | resnext50_32x4d, resnext101_32x8d | 224×224 | ResNet 升级版 |
| **EfficientNet** | efficientnet_b0~b7 | 224~600 | 效率优先 |
| **EfficientNetV2** | efficientnet_v2_s/m/l | 224~480 | 训练更快 |
| **MobileNet** | mobilenet_v2, mobilenet_v3_large/small | 224×224 | 移动端优化 |
| **ShuffleNet** | shufflenet_v2_x0_5/x1_0/x1_5/x2_0 | 224×224 | 计算高效 |
| **SqueezeNet** | squeezenet1_0, squeezenet1_1 | 224×224 | 参数极少 |
| **MNASNet** | mnasnet0_5/0_75/1_0/1_3 | 224×224 | 移动端 NAS |
| **DenseNet** | densenet121/161/169/201 | 224×224 | 密集连接 |
| **VGG** | vgg11/13/16/19 | 224×224 | 经典但参数大 |
| **ConvNeXt** | convnext_tiny/small/base/large | 224×224 | 现代 CNN |
| **ViT** | vit_b_16, vit_b_32, vit_l_16 | 224×224 | Vision Transformer |

### 3.2 torchvision 版本兼容性

| torchvision 版本 | 权重加载方式 | 代码差异 |
|-----------------|-------------|---------|
| < 0.13 | `model(pretrained=True)` | 直接参数 |
| ≥ 0.13 | `model(weights=WeightsEnum.IMAGENET1K_V1)` | 需构造 Weights 类 |

0.13+ 版本中，权重类名遵循 `{ModelName}_Weights` 约定，例如：
- `resnet18` → `ResNet18_Weights.IMAGENET1K_V1`
- `efficientnet_b0` → `EfficientNet_B0_Weights.IMAGENET1K_V1`

### 3.3 预处理对齐

分类模型的预处理必须与训练时的规范一致，这是 ONNX 推理精度正确的关键。

**ImageNet 标准预处理：**

```
OpenCV (BGR) ──▶ BGR→RGB ──▶ Resize(256) ──▶ CenterCrop(224) ──▶ Normalize(mean, std) ──▶ HWC→CHW
```

| 步骤 | 参数 | 说明 |
|------|------|------|
| Color Convert | BGR → RGB | OpenCV 默认 BGR，需转换 |
| Resize | 256 (短边) | 保持宽高比缩放 |
| CenterCrop | 224×224 | 从中心裁剪 |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet 统计值 |
| Layout | HWC → CHW | 通道维前置 |
| Batch | 添加 batch 维度 → (1, 3, H, W) | 单样本推理 |

### 3.4 输入输出规格

| 项目 | 规范 |
|------|------|
| 输入名 | `image` |
| 输入 shape | `(1, 3, H, W)`，H/W 取决于模型（默认 224） |
| 输入 dtype | `float32` |
| 输入范围 | Normalize 后，均值 0、方差 1 |
| 输出名 | `output0` |
| 输出 shape | `(1, num_classes)`，通常 1000 |
| 输出含义 | logits（未经过 softmax） |

### 3.5 使用方式

```bash
# torchvision 预训练模型（自动下载权重）
python3 -m export.onnx.convert \
    --model efficientnet_b0 \
    --save models/runtime/efficientnet_b0.onnx \
    --img-size 224
```

## 4. Ultralytics 模型导出

### 4.1 任务类型

| 任务 | 模型后缀 | 输出结构 | 典型模型 |
|------|---------|---------|---------|
| Detection | `yolov8n` / `yolo11n` | `(1, 84, 8400)` | yolov8s, yolo11m |
| Segmentation | `yolov8n-seg` | `(1, 116, 8400)` + proto mask | yolov8s-seg |
| Classification | `yolov8n-cls` | `(1, num_classes)` | yolov8s-cls |
| Pose | `yolov8n-pose` | `(1, 56, 8400)` | yolov8s-pose |

**输出结构说明（以 Detection 为例）：**

```
pred shape: (1, 84, 8400)
    ├── 84 = 4 (bbox) + 80 (COCO classes)
    └── 8400 = 3 个检测头的 grid 点总数 (80×80 + 40×40 + 20×20)
```

### 4.2 YOLOv8 → YOLO11 演进

| 特性 | YOLOv8 | YOLO11 | 影响 |
|------|--------|--------|------|
| 输出结构 | 单输出 `(1, 84+n, N)` | 同上 | 模型无关 |
| 预处理 | LetterBox → 640×640 | 同上 | 兼容 |
| 导出接口 | `model.export(format="onnx")` | 同上 | 兼容 |
| 后处理 | 已对齐 | 已对齐 | 兼容 |

YOLO 系列模型的输出结构在 v8 之后趋于稳定，v8→v11 的导出和后处理**完全兼容**。未来版本（如 YOLO26）预计仍保持相似的输出格式。

### 4.3 预处理对齐

**Ultralytics 模型统一预处理规范：**

```
OpenCV (BGR) ──▶ LetterBox(640) ──▶ BGR→RGB ──▶ Normalize(÷255) ──▶ HWC→CHW
```

| 步骤 | 参数 | 说明 |
|------|------|------|
| LetterBox | target_size=640, auto_pad=True | 保持宽高比，填充至正方形 |
| Color Convert | BGR → RGB | 颜色通道转换 |
| Normalize | 除以 255.0 | 缩放到 [0, 1] |
| Layout | HWC → CHW | 通道维前置 |
| Batch | 添加 batch 维度 → `(1, 3, 640, 640)` | 单样本推理 |

**LetterBox 的注意点：**
- LetterBox 会记录填充信息 `(pad_w, pad_h)`，后处理时需要**逆填充**还原框坐标
- 不同实现（NumPy / PyTorch）的填充值须保持一致

### 4.4 输入输出规格

| 项目 | 规范 |
|------|------|
| 输入名 | `image`（Ultralytics 默认名） |
| 输入 shape | `(1, 3, 640, 640)` |
| 输入 dtype | `float32` |
| 输入范围 | `[0, 1]`（÷255 后） |
| 输出名 | `output0` |
| 输出含义 | 检测头原始输出（未后处理） |

### 4.5 使用方式

```bash
# Ultralytics 导出（自动下载预训练权重）
python3 -m export.onnx.ultralytics \
    --weights yolov8s \
    --save models/runtime/yolov8s.onnx \
    --opset 12

# 分割模型
python3 -m export.onnx.ultralytics \
    --weights yolov8s-seg \
    --save models/runtime/yolov8s-seg.onnx
```

## 5. torchvision vs Ultralytics 导出对比

| 对比项 | torchvision | Ultralytics |
|--------|------------|-------------|
| 模型类型 | 分类 | 检测、分割、分类、姿态 |
| 导出方式 | `torch.onnx.export` 手动封装 | `YOLO(model).export(format="onnx")` |
| 预处理 | Resize+Crop+Normalize | LetterBox+÷255 |
| 后处理 | softmax → top-k | NMS → 解码 |
| ONNX 输入名 | `image` | `image`（可配置） |
| 动态 batch | 通过 `dynamic_axes` 支持 | 由 Ultralytics 内部处理 |

## 6. ONNX 验证

### 6.1 格式合法性检查

```python
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
```

### 6.2 PT vs ONNX 输出对比

导出后必须进行数值对比，确保 ONNX 输出与 PyTorch 一致：

| 检查项 | 方法 | 接受标准 |
|--------|------|---------|
| PT vs ONNX (FP32) | 同一随机输入 | `rtol=1e-3, atol=1e-5` |
| 多输出匹配 (分割) | 逐输出对比 | 同上 |

```python
# 伪代码流程
input_tensor = torch.randn(1, 3, 640, 640)
torch_output = model(input_tensor)
onnx_output = ort_session.run(None, {"image": input_tensor.numpy()})
np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-5)
```

## 7. ONNX 优化

导出后的 ONNX 模型可进一步优化，提升推理速度或减小体积。

### 7.1 常量折叠（已在导出时默认开启）

`do_constant_folding=True` 会将导出时能计算出的常量节点折叠为定值。

### 7.2 ONNX Simplifier

使用 `onnx-simplifier` 简化计算图：

```bash
python3 -m onnxsim input.onnx output.onnx
```

简化内容包括：
- 移除冗余 Identity 节点
- 合并相邻的 Reshape/Transpose
- 消除无用 Cast 操作

### 7.3 适用时机

| 优化类型 | 位置 | 推荐 |
|---------|------|------|
| 常量折叠 | `torch.onnx.export` 参数 | ✅ 默认开启 |
| onnx-simplifier | 导出后可选 | ✅ 建议执行 |
