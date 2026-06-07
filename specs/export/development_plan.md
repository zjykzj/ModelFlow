# Export 模块开发计划

> **目标：** 在 `export2/` 目录下按 specs 定义全新实现模型导出模块
> **原则：** 零外部依赖（不引用项目根目录 `core/`）、预处理自包含、模块化

---

## 一、开发阶段总览

```
阶段 1: 基础设施 (core/)      ← 无依赖，最先构建
    │
    ▼
阶段 2: ONNX 导出 (onnx/)    ← 依赖 core/
    │
    ▼
阶段 3: TensorRT (tensorrt/) ← 依赖 core/（前置条件：ONNX 文件）
    │
    ▼
阶段 4: Triton 配置 (triton/)← 依赖 core/（前置条件：ONNX/TRT 文件）
    │
    ▼
阶段 5: 辅助脚本 (scripts/)  ← 依赖 core/ 提供的预处理工具
    │
    ▼
阶段 6: 测试 (tests/)        ← 依赖所有模块
```

**阶段间依赖关系：** 每个阶段可独立开发，但测试需前置阶段完成。

---

## 二、阶段 1：基础设施 `core/`

### 文件清单

| 文件 | 职责 | 参考现有代码 |
|------|------|------------|
| `core/__init__.py` | 包初始化，导出关键接口 | — |
| `core/base.py` | `BaseExporter` 抽象基类，统一所有导出器的接口契约 | 新设计 |
| `core/validation.py` | ONNX 格式校验、PT vs ONNX/TRT 输出对比 | `pytorch_to_onnx.py` 中的 `check_onnx()` / `check_output()` |
| `core/utils.py` | 公共工具：预处理工具（LetterBox、Resize、Normalize）、文件操作 | 新设计（替代原 `core.npy.*`） |

### 设计决策

#### A. `BaseExporter` 接口

```python
class BaseExporter(ABC):
    """所有导出器的统一基类"""

    @abstractmethod
    def export_onnx(self, output_path: str, **kwargs) -> str:
        """导出 ONNX 模型，返回 ONNX 路径"""
        ...

    def validate_onnx(self, onnx_path: str) -> bool:
        """验证 ONNX 合法性（默认实现调用 validation.py）"""
        ...

    def compare_output(self, onnx_path: str, *args, **kwargs) -> bool:
        """对比 PyTorch 与 ONNX 输出（默认实现调用 validation.py）"""
        ...
```

#### B. 预处理工具（核心设计）

**为什么自包含：** 设计约束禁止引用项目根目录 `core/`，所以必须自己实现。

```
core/utils.py
├── PreprocessFactory          # 预处理工厂
│   ├── create_classify_pipeline()  # 分类预处理 (Resize+CenterCrop+Normalize)
│   └── create_detect_pipeline()    # 检测预处理 (LetterBox+÷255)
├── letterbox(image, target_size)   # LetterBox 函数
├── classify_preprocess(image, ...) # 分类预处理完整管线
└── detect_preprocess(image, ...)   # 检测预处理完整管线
```

**同时服务于：**
- `scripts/generate_calib_cache_for_*.py` 的校准数据生成
- 用户自行验证模型预处理效果

---

## 三、阶段 2：ONNX 导出 `onnx/`

### 文件清单

| 文件 | 职责 | 参考现有代码 |
|------|------|------------|
| `onnx/__init__.py` | 包初始化 | — |
| `onnx/convert.py` | torchvision 分类模型导出（支持所有主流模型） | `pytorch_to_onnx.py` |
| `onnx/ultralytics.py` | Ultralytics 检测/分割/分类/姿态模型导出 | `ultarlytics_export.py` |
| `onnx/optimize.py` | ONNX 图优化（onnx-simplifier 封装） | 新开发 |

### 设计决策

#### A. torchvision 导出（`convert.py`）

- 支持模型列表：ResNet, EfficientNet, MobileNet, ShuffleNet, SqueezeNet, MNASNet, DenseNet, VGG, ConvNeXt, ViT
- 版本兼容：自动检测 torchvision 版本，选择 weights 加载方式
- 输入配置：可配置 `--img-size`、`--img-channels`
- 自动验证：导出后自动执行 PT vs ONNX 输出对比

#### B. Ultralytics 导出（`ultralytics.py`）

- 支持 YOLOv8 / YOLO11 / 未来版本（通过 `YOLO()` 自动兼容）
- 任务类型：detect / segment / classify / pose
- opset 策略：默认 opset=12，可指定

#### C. ONNX 优化（`optimize.py`）

- `onnx-simplifier` 封装
- 主要操作：移除冗余节点、融合 Transpose、消除无用 Cast

---

## 四、阶段 3：TensorRT `tensorrt/`

### 文件清单

| 文件 | 职责 | 参考现有代码 |
|------|------|------------|
| `tensorrt/__init__.py` | 包初始化 | — |
| `tensorrt/calibrator.py` | INT8 校准器基类 + 公用工具 | 新设计 |
| `tensorrt/build_fp16.py` | FP16 引擎构建（封装 trtexec / TensorRT Python API） | 新开发 |
| `tensorrt/build_int8.py` | INT8 引擎构建（PyTorch 校准器） | `safe_int8_build_by_torch.py` |
| `tensorrt/build_int8_pycuda.py` | INT8 引擎构建（PyCUDA 校准器，Jetson/嵌入式） | `safe_int8_build_by_pycuda.py` |

### 设计决策

#### A. 校准器层次

```
calibrator.py
├── BaseCalibrator(trt.IInt8EntropyCalibrator2)  # 通用逻辑
├── TorchCalibrator(BaseCalibrator)               # PyTorch 版
└── PyCudaCalibrator(BaseCalibrator)              # PyCUDA 版
```

`BaseCalibrator` 中提取：
- 文件扫描、过滤、排序
- NaN/Inf 数据自愈
- 进度日志
- 缓存读写

两版各自实现数据传输逻辑：
- `TorchCalibrator`：`torch.pin_memory` + `cuda.copy_`
- `PyCudaCalibrator`：`pycuda.pagelocked_empty` + `memcpy_htod`

#### B. FP16 构建方式

两种构建策略：
1. **trtexec 封装**（默认）—— 通过 `subprocess` 调用 trtexec
2. **TensorRT Python API**（备选）—— 编程式构建

优先实现 trtexec 封装，简单可靠。

---

## 五、阶段 4：Triton 配置 `triton/`

### 文件清单

| 文件 | 职责 | 参考现有代码 |
|------|------|------------|
| `triton/__init__.py` | 包初始化 | — |
| `triton/config_generator.py` | `config.pbtxt` 自动生成 | 新开发 |
| `triton/model_repo.py` | 仓库目录结构生成、模型文件管理 | 新开发 |

### 功能范围

- 支持 ONNX Runtime 和 TensorRT 两种后端
- 任务感知（detect / classify / segment / pose）→ 自动设置输入输出 dims
- 支持 dynamic batching 配置
- 支持版本管理（`version_policy`）
- 模型命名规范自动应用

---

## 六、阶段 5：辅助脚本 `scripts/`

### 文件清单

| 文件 | 职责 | 需重写？ | 原因 |
|------|------|:--------:|------|
| `scripts/generate_calib_cache_for_coco.py` | 检测模型 INT8 校准数据生成 | **是** | 原来依赖 `core.npy.yolov8_preprocess` |
| `scripts/generate_calib_cache_for_imagenet.py` | 分类模型 INT8 校准数据生成 | **是** | 原来依赖 `core.npy.classify_preprocess` |
| `scripts/random_copy_images.py` | 从数据集随机抽样图片 | **否** | 无外部依赖 |

**重写策略：** 两个校准脚本改用 `export2/core/utils.py` 提供的预处理函数，实现零外部依赖。

---

## 七、阶段 6：测试 `tests/`

### 文件清单

| 文件 | 职责 |
|------|------|
| `tests/__init__.py` | 包初始化 |
| `tests/test_export.py` | 测试 torchvision 和 Ultralytics 的 ONNX 导出 |
| `tests/test_engine.py` | 测试 TensorRT 引擎构建与推理（需要 GPU） |

### 测试策略

| 测试类型 | 覆盖范围 | 环境要求 |
|---------|---------|---------|
| **单元测试** | `core/` 工具函数、预处理 | 无 GPU |
| **导出测试** | PT→ONNX 导出 + 输出对比 | PyTorch, onnxruntime |
| **引擎测试** | TensorRT 构建 + 加载 + 推理 | GPU + TensorRT |

---

## 八、代码复用策略

| 现有文件 | 复用方式 | 需要改动 |
|---------|---------|---------|
| `pytorch_to_onnx.py` | 逻辑复制到 `onnx/convert.py` | 提取验证逻辑到 `core/validation.py`；提取预处理到 `core/utils.py` |
| `ultarlytics_export.py` | 逻辑复制到 `onnx/ultralytics.py` | 精简下载 URL 映射（`YOLO()` 已自动下载） |
| `safe_int8_build_by_torch.py` | 逻辑复制到 `tensorrt/build_int8.py` | 拆分校准器到 `calibrator.py` |
| `safe_int8_build_by_pycuda.py` | 逻辑复制到 `tensorrt/build_int8_pycuda.py` | 同上 |
| `scripts/generate_calib_cache_*.py` | 结构保留 | 改用 `export2/core/utils.py` 的预处理函数 |
| `scripts/random_copy_images.py` | 直接复制 | 基本无需改动 |

---

## 九、交付清单

| 阶段 | 文件数 | 产出 |
|------|:-----:|------|
| 1. core/ | 4 | 基类、验证工具、预处理工具 |
| 2. onnx/ | 4 | torchvision 导出、Ultralytics 导出、ONNX 优化 |
| 3. tensorrt/ | 5 | 校准器基类、FP16/INT8 构建器 |
| 4. triton/ | 3 | config 生成器、仓库管理 |
| 5. scripts/ | 3 | 校准数据准备、图片抽样 |
| 6. tests/ | 3 | 导出测试、引擎测试 |
| **合计** | **22** | **完整导出管线** |
