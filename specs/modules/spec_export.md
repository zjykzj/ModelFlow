# 模型导出模块规格

> **Version:** 0.1
> **Status:** Draft
> **Dependencies:** `spec_architecture.md`（独立模块设计原则）

## 1. 模块定位

`export/` 是一个**独立模块**，不依赖 `modelflow/`、`cpp/` 或项目根目录下 `core/` 的任何代码。职责是将 PyTorch 模型转换为 ONNX，再进一步转换为 TensorRT 引擎，以及生成 Triton 模型仓库配置。

```
PyTorch (.pt) ──▶ ONNX (.onnx) ──▶ TensorRT (.engine) FP16 / INT8
                                      │
                                      ▼
                               Triton 模型仓库 (config.pbtxt + model)
```

**知识层参考：** 格式理解、转换原理、模型差异和选型依据详见 [`specs/export/`](../export/index.md) 系列文档。

### 1.1 设计约束

| # | 约束 | 说明 | 违反示例 |
|---|------|------|---------|
| 1 | **零外部依赖** | `export/` 下的代码不得 import 项目根目录 `core/`、`modelflow/`、`cpp/` 或任何其他模块 | ❌ `from core.npy.yolov8_preprocess import ImgPrepare` |
| 2 | **预处理自包含** | 校准数据准备脚本（`scripts/`）中的预处理逻辑必须自实现，或使用 `export/core/` 提供的工具 | |
| 3 | **依赖版本约束** | 允许的第三方依赖：`torch`、`onnx`、`onnxruntime`、`tensorrt` (>=10.0)、`pycuda`（可选）、`ultralytics`（可选） | |
| 4 | **可选拔件懒加载** | `tensorrt`、`pycuda`、`ultralytics` 等可选依赖不得在模块级别 `import`，必须在使用处按需导入并捕获 `ImportError` 给出安装指引 | ❌ 模块级 `import tensorrt as trt` 阻断不依赖 tensorrt 的子模块（如 `build_fp16` 无法在无 TRT 环境运行） |

**背景：** 此约束确保 `export/` 模块可独立开发、独立测试，不因项目其他模块的变更而阻塞。当需要复用在 `modelflow/` 或项目 `core/` 中已有的预处理逻辑时，应将其**复制或重构**到 `export/core/` 中，而非直接引用原位置代码。

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
│   ├── build_fp16.py               # FP16 构建器
│   ├── build_int8.py               # INT8 构建器（PyTorch 校准器）
│   ├── build_int8_pycuda.py        # INT8 构建器（PyCUDA 校准器，用于 Jetson）
│   └── calibrator.py               # INT8 校准器基类
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

**结构说明：**

| 子模块 | 职责 | 知识层文档 |
|--------|------|-----------|
| `core/` | 公共基础设施：基类、验证、工具函数 | — |
| `onnx/` | PT→ONNX 导出 | [`specs/export/onnx_export.md`](../export/onnx_export.md) |
| `tensorrt/` | ONNX→TensorRT 引擎构建 | [`specs/export/tensorrt_conversion.md`](../export/tensorrt_conversion.md) |
| `triton/` | Triton 配置生成 | [`specs/export/triton_deployment.md`](../export/triton_deployment.md) |
| `scripts/` | 校准数据准备等辅助脚本 | — |
| `tests/` | 导出测试和引擎测试 | — |

## 3. 导出管线

### 3.1 管线阶段

```
Stage 1: PT → ONNX
    ├── torchvision 模型: torch.onnx.export 手动封装
    └── Ultralytics 模型: YOLO().export(format="onnx")
            │
            ▼
Stage 2: ONNX → TensorRT  (可选)
    ├── FP16: 快速精度无损，无需校准数据
    └── INT8: 需校准数据和数据集预处理
            │
            ▼
Stage 3: Triton 配置   (可选)
    └── 生成 config.pbtxt + 模型仓库结构
```

### 3.2 导出深度

| 等级 | 产出 | 适用场景 | 命令参考 |
|------|------|---------|---------|
| **L1** | `.onnx` | ONNX Runtime 推理 | `export/onnx/convert.py` 或 `export/onnx/ultralytics.py` |
| **L2** | `.onnx` + `.engine` | TensorRT GPU 推理 | L1 + `export/tensorrt/build_fp16.py` 或 `build_int8.py` |
| **L3** | `.onnx` + `.engine` + Triton 配置 | Triton Server 部署 | L1/L2 + `export/triton/config_generator.py` |

## 4. 任务导出规范

### 4.1 模型来源与任务

| 模型来源 | 任务类型 | L1 | L2 | L3 |
|---------|---------|:--:|:--:|:--:|
| torchvision | Classification | ✅ | ✅ | ✅ |
| Ultralytics | Detection | ✅ | ✅ | ✅ |
| Ultralytics | Segmentation | ✅ | ✅ | ✅ |
| Ultralytics | Classification | ✅ | ✅ | ✅ |
| Ultralytics | Pose | ✅ | ✅ | ✅ |

### 4.2 输入输出规格

| 任务 | 输入 shape | 输入名 | 输出说明 |
|------|-----------|--------|---------|
| Classification | `(1, 3, 224, 224)` | `image` | logits `(1, num_classes)` |
| Detection | `(1, 3, 640, 640)` | `image` | pred `(1, 84, 8400)` |
| Segmentation | `(1, 3, 640, 640)` | `image` | pred `(1, 116, 8400)` + proto mask |
| Pose | `(1, 3, 640, 640)` | `image` | pred `(1, 56, 8400)` |

## 5. 精度验证

| 检查项 | 方法 | 标准 |
|--------|------|------|
| ONNX 合法性 | `onnx.checker.check_model()` | 不报错 |
| PT vs ONNX | 同一输入对比输出 | rtol=1e-3, atol=1e-5 |
| FP32 vs FP16 | 同一输入对比输出 | FP16 精度损失 < 0.1% |
| FP32 vs INT8 | 同一输入对比输出 | INT8 精度损失 < 1% |
