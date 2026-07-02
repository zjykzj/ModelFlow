# ONNX Export Specification

> **Status:** Implemented
> **Version:** 0.1
> **Prerequisite Reading:** `specs/modules/spec_export.md` (export module architecture), `specs/export/index.md` (export knowledge layer overview)

## 1. Overview

ONNX (Open Neural Network Exchange) is the **first stop** in the PyTorch model export pipeline and a prerequisite for subsequent TensorRT conversion and Triton deployment. As an intermediate representation (IR), ONNX provides:

| Feature | Description |
|------|------|
| **Framework-agnostic** | Not tied to PyTorch; loadable from C++/Python/JS |
| **Standardized operators** | Standard operator set defined via ONNX opset |
| **Optimizable** | Supports constant folding, node fusion, and other graph optimizations |
| **Quantizable** | INT8/FP16 quantization both build on the ONNX computation graph |

## 2. Export Mechanism

### 2.1 How `torch.onnx.export` Works

PyTorch exports to ONNX using a **Tracing** mechanism:

```
Sample input tensor ──▶ Model forward pass ──▶ Record actually executed operators ──▶ ONNX computation graph
```

```python
torch.onnx.export(
    model=model,              # PyTorch model
    args=dummy_input,         # Sample input (determines input shape and trace path)
    f=model.onnx,             # Output path
    input_names=["image"],    # Input node naming
    output_names=["output0"], # Output node naming
    opset_version=12,         # ONNX opset version
    dynamic_axes=None,        # Dynamic axis configuration
    do_constant_folding=True  # Constant folding optimization
)
```

**Limitations of Tracing:**
- Only records branches that are **actually taken** — if the model contains `if` statements whose conditions depend on input, the other branch will not appear in the ONNX graph
- Control flow (loops, conditionals) is unrolled into a static graph
- Models with dynamic shapes (e.g., variable input sizes) require the `dynamic_axes` parameter

### 2.2 Opset Version

The ONNX opset defines the set of available operators; higher versions support more operators:

| opset | PyTorch Version | Notes |
|-------|-------------|------|
| 11 | 1.5+ | Broad support, good compatibility |
| **12** | 1.6+ | ✅ **Recommended default**, balances operator support and compatibility |
| 13 | 1.8+ | New hardware-aware operators |
| 14 | 1.10+ | Additional operators |
| 15 | 1.11+ | More quantization operator support |
| 16+ | 1.12+ | Ongoing evolution |

**Selection principles:**
- **Default to opset=12**, the most robust choice at present
- TensorRT 10.x has good support for opset 12
- If you encounter unsupported operators, try upgrading the opset version
- A higher opset is not necessarily better — it may reduce compatibility with the target backend

## 3. Exporting torchvision Classification Models

### 3.1 Supported Model List

| Family | Models | Default Input Size | Notes |
|------|------|-------------|------|
| **ResNet** | resnet18, resnet34, resnet50, resnet101, resnet152 | 224x224 | Classic classification backbone |
| **ResNeXt** | resnext50_32x4d, resnext101_32x8d | 224x224 | Upgraded ResNet |
| **EfficientNet** | efficientnet_b0~b7 | 224~600 | Efficiency-first |
| **EfficientNetV2** | efficientnet_v2_s/m/l | 224~480 | Faster training |
| **MobileNet** | mobilenet_v2, mobilenet_v3_large/small | 224x224 | Mobile-optimized |
| **ShuffleNet** | shufflenet_v2_x0_5/x1_0/x1_5/x2_0 | 224x224 | Computationally efficient |
| **SqueezeNet** | squeezenet1_0, squeezenet1_1 | 224x224 | Minimal parameters |
| **MNASNet** | mnasnet0_5/0_75/1_0/1_3 | 224x224 | Mobile NAS |
| **DenseNet** | densenet121/161/169/201 | 224x224 | Dense connectivity |
| **VGG** | vgg11/13/16/19 | 224x224 | Classic but large parameter count |
| **ConvNeXt** | convnext_tiny/small/base/large | 224x224 | Modern CNN |
| **ViT** | vit_b_16, vit_b_32, vit_l_16 | 224x224 | Vision Transformer |

### 3.2 torchvision Version Compatibility

| torchvision Version | Weight Loading | Code Difference |
|-----------------|-------------|---------|
| < 0.13 | `model(pretrained=True)` | Direct parameter |
| >= 0.13 | `model(weights=WeightsEnum.IMAGENET1K_V1)` | Requires Weights class construction |

In version 0.13+, the weight class name follows the `{ModelName}_Weights` convention, for example:
- `resnet18` -> `ResNet18_Weights.IMAGENET1K_V1`
- `efficientnet_b0` -> `EfficientNet_B0_Weights.IMAGENET1K_V1`

### 3.3 Preprocessing Alignment

Classification model preprocessing must match the training convention exactly — this is critical for correct ONNX inference accuracy.

**ImageNet standard preprocessing:**

```
OpenCV (BGR) ──▶ BGR->RGB ──▶ Resize(256) ──▶ CenterCrop(224) ──▶ Normalize(mean, std) ──▶ HWC->CHW
```

| Step | Parameters | Notes |
|------|------|------|
| Color Convert | BGR -> RGB | OpenCV uses BGR by default; conversion needed |
| Resize | 256 (short side) | Scale preserving aspect ratio |
| CenterCrop | 224x224 | Crop from center |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet statistics |
| Layout | HWC -> CHW | Channel dimension first |
| Batch | Add batch dimension -> (1, 3, H, W) | Single-sample inference |

### 3.4 Input/Output Specification

| Item | Spec |
|------|------|
| Input name | `image` |
| Input shape | `(1, 3, H, W)`, H/W depends on model (default 224) |
| Input dtype | `float32` |
| Input range | Post-normalization, zero mean, unit variance |
| Output name | `output0` |
| Output shape | `(1, num_classes)`, typically 1000 |
| Output meaning | logits (not passed through softmax) |

### 3.5 Usage

Export via `export.onnx.convert` module. Accepts `--model` (torchvision model name), `--save` (output path), and `--img-size` (input resolution). Weights are auto-downloaded.

## 4. Exporting Ultralytics Models

### 4.1 Task Types

| Task | Model Suffix | Output Structure | Typical Models |
|------|---------|---------|---------|
| Detection (v5) | `yolov5s` | `(1, 25200, 85)` | yolov5s, yolov5m |
| Detection (v8/v11) | `yolov8n` / `yolo11n` | `(1, 84, 8400)` | yolov8s, yolo11m |
| Segmentation | `yolov8n-seg` | `(1, 116, 8400)` + proto mask | yolov8s-seg |
| Classification | `yolov8n-cls` | `(1, num_classes)` | yolov8s-cls |
| Pose | `yolov8n-pose` | `(1, 56, 8400)` | yolov8s-pose |

**Output structure explained (using Detection as an example):**

```
pred shape: (1, 84, 8400)
    |-- 84 = 4 (bbox) + 80 (COCO classes)
    |-- 8400 = total grid points across 3 detection heads (80x80 + 40x40 + 20x20)
```

### 4.2 YOLOv8 -> YOLO11 Evolution

| Feature | YOLOv8 | YOLO11 | Impact |
|------|--------|--------|------|
| Output structure | Single output `(1, 84+n, N)` | Same as v8 | Model-agnostic |
| Preprocessing | LetterBox -> 640x640 | Same as v8 | Compatible |
| Export interface | `model.export(format="onnx")` | Same as v8 | Compatible |
| Postprocessing | Already aligned | Already aligned | Compatible |

Export and postprocessing are **fully compatible** between v8 and v11. YOLOv5 uses a different output shape (`(1, 25200, 85)` vs `(1, 84, 8400)`) and requires the `model_version="v5"` parameter for correct postprocessing.

### 4.3 Preprocessing Alignment

**Unified preprocessing specification for Ultralytics models:**

```
OpenCV (BGR) ──▶ LetterBox(640) ──▶ BGR->RGB ──▶ Normalize(/255) ──▶ HWC->CHW
```

| Step | Parameters | Notes |
|------|------|------|
| LetterBox | target_size=640, auto_pad=True | Preserve aspect ratio, pad to square |
| Color Convert | BGR -> RGB | Color channel conversion |
| Normalize | Divide by 255.0 | Scale to [0, 1] |
| Layout | HWC -> CHW | Channel dimension first |
| Batch | Add batch dimension -> `(1, 3, 640, 640)` | Single-sample inference |

**LetterBox caveats:**
- LetterBox records padding information `(pad_w, pad_h)`; postprocessing must apply **inverse padding** to restore box coordinates
- Padding values must be consistent across different implementations (NumPy / PyTorch)

### 4.4 Input/Output Specification

| Item | Spec |
|------|------|
| Input name | `image` (Ultralytics default name) |
| Input shape | `(1, 3, 640, 640)` |
| Input dtype | `float32` |
| Input range | `[0, 1]` (after /255) |
| Output name | `output0` |
| Output meaning | Raw detection head output (not postprocessed) |

### 4.5 Usage

Export via `export.onnx.ultralytics` module. Accepts model name (e.g., `yolov8s`, `yolov8s-seg`) and `--save` for output path. Weights are auto-downloaded.

## 5. torchvision vs Ultralytics Export Comparison

| Item | torchvision | Ultralytics |
|--------|------------|-------------|
| Model type | Classification | Detection, segmentation, classification, pose |
| Export method | `torch.onnx.export` manual wrapper | `YOLO(model).export(format="onnx")` |
| Preprocessing | Resize+Crop+Normalize | LetterBox+/255 |
| Postprocessing | softmax -> top-k | NMS -> decode |
| ONNX input name | `image` | `image` (configurable) |
| Dynamic batch | Supported via `dynamic_axes` | Handled internally by Ultralytics |

## 6. ONNX Validation

### 6.1 Format Validity Check

```python
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
```

### 6.2 PT vs ONNX Output Comparison

After export, a numerical comparison is mandatory to ensure the ONNX output matches PyTorch:

| Check | Method | Acceptance Criteria |
|--------|------|---------|
| PT vs ONNX (FP32) | Same random input | `rtol=1e-3, atol=1e-5` |
| Multi-output matching (segmentation) | Compare output by output | Same as above |

## 7. ONNX Optimization

The exported ONNX model can be further optimized to improve inference speed or reduce size.

### 7.1 Constant Folding (Already Enabled by Default During Export)

`do_constant_folding=True` folds constant nodes that can be computed at export time into fixed values.

### 7.2 ONNX Simplifier

Use `onnx-simplifier` to simplify the computation graph:

```bash
python3 -m onnxsim input.onnx output.onnx
```

Simplification includes:
- Removing redundant Identity nodes
- Merging adjacent Reshape/Transpose operations
- Eliminating unnecessary Cast operations

### 7.3 When to Apply

| Optimization Type | Where | Recommended |
|---------|------|------|
| Constant folding | `torch.onnx.export` parameter | ✅ Enabled by default |
| onnx-simplifier | After export (optional) | ✅ Recommended |
