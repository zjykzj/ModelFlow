# export/ — 模型导出管线

> 📖 **详细文档已迁移至 [`docs/export/`](../docs/export/index.md)**，提供按模型类型和导出深度的完整指南。

## 快速导航

| 我要... | 看这篇 |
|---------|--------|
| 快速上手导出 | [导出总览与快速开始](../docs/export/index.md) |
| 导出分类模型 (ResNet/MobileNet/EfficientNet) | [分类模型导出](../docs/export/classification.md) |
| 导出检测模型 (YOLOv8/YOLO11/YOLO26) | [检测模型导出](../docs/export/detection.md) |
| 导出分割模型 (YOLOv8-seg) | [分割模型导出](../docs/export/segmentation.md) |
| 了解 TensorRT FP16/INT8 量化 | [TensorRT 深入指南](../docs/export/tensorrt.md) |
| 配置 Triton 部署 | [Triton 部署指南](../docs/export/triton.md) |

## 最小示例

```bash
# L1: PT → ONNX
python3 -m export.onnx.convert --model efficientnet_b0 --save model.onnx
python3 -m export.onnx.ultralytics yolov8s --save yolov8s.onnx

# L2: ONNX → TensorRT
python3 -m export.tensorrt.build_fp16 --onnx model.onnx --save model_fp16.engine

# L3: Triton 配置
python3 -m export.triton.config_generator \
    --model-name Detect_COCO_YOLOv8s_TRT \
    --backend tensorrt --task detect --save ./models/triton/
```

## 项目结构

```
export/
├── core/           # 基础设施（BaseExporter、验证、预处理）
├── onnx/           # PT → ONNX 导出（torchvision + Ultralytics）
├── tensorrt/       # ONNX → TensorRT 引擎构建（FP16 / INT8）
├── triton/         # Triton 配置生成 + 模型仓库管理
├── scripts/        # 辅助脚本（校准数据生成）
└── tests/          # 测试
```
