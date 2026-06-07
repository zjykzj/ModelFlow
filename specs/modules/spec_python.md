# Python 模块规格

> **Version:** 0.1
> **Status:** Draft
> **Dependencies:** `spec_architecture.md` (Pipeline 模式、三层模块)

## 1. 目录结构

```
modelflow/
├── __init__.py
├── core/                   # 基础设施
│   ├── interfaces.py       # 抽象基类
│   ├── registry.py         # 注册机制
│   ├── types.py            # 枚举 + ModelInfo
│   └── config.py           # 配置管理
├── cfgs/                   # 数据集类别配置
│   ├── coco.py             # COCO 类别列表（detect/segment）
│   └── imagenet.py         # ImageNet 类别列表（classify）
├── backends/               # 推理后端
│   ├── base.py             # BaseBackend（含 task_type/class_list）
│   ├── onnx.py             # OnnxBackend
│   ├── tensorrt.py         # TensorrtBackend
│   └── triton.py           # TritonBackend
├── processors/             # 预处理 + 后处理
│   ├── base.py             # BasePreprocessor, BasePostprocessor
│   ├── classify/
│   ├── detect/
│   ├── segment/
│   ├── semantic_seg/
│   └── multimodal/
├── pipelines/              # 预构建 Pipeline
│   └── *_pipeline.py
├── datasets/               # 数据集
│   └── *.py
├── evaluators/             # 评估编排（benchmark/eval 脚本调用此层）
│   └── *.py
├── metrics/                # 指标
│   └── *.py
├── viz/                    # 可视化
│   └── *.py
└── utils/                  # 工具
    ├── logger.py
    ├── profile.py
    └── helpers.py
```

## 2. 核心接口

### 2.1 InferencePipeline

```python
class InferencePipeline:
    """
    推理管线 = Preprocessor + Backend + Postprocessor

    完整的数据流：
        image (HWC, BGR) → Preprocessor → tensor (NCHW) → Backend → raw (List[ndarray])
            ├── __call__: raw → Postprocessor → StructuredResult (dict)
            └── infer:     仅返回 raw（评估场景用，Evaluator 自行后处理）

    用法:
        pipeline = InferencePipeline(preprocessor, backend, postprocessor)
        result = pipeline(image, conf_thres=0.25, iou_thres=0.45)  # 端到端推理
        raw = pipeline.infer(tensor)  # 仅后端推理（评估循环内使用）
    """
    def __init__(self, preprocessor, backend, postprocessor): ...
    def __call__(self, image, **kwargs) -> Any: ...
    def infer(self, tensor: np.ndarray) -> List[np.ndarray]: ...
    def warmup(self): ...
```

**Pipeline 生命周期：**

```
Evaluator
    │
    ├── 迭代 Dataset（每张图）
    │   ├── preprocessor(image)           → tensor
    │   ├── backend(tensor)               → raw List[ndarray]
    │   ├── postprocessor(raw, **kwargs)  → StructuredResult
    │   └── metrics.update(result, ground_truth)
    │
    └── metrics.compute() → Dict[str, float]
```

### 2.2 BaseBackend

后端的职责是**纯张量推理**，从预处理好的张量到原始输出。它需要知道任务类型和类别列表以正确解释输出：

```python
class BaseBackend(ABC):
    """推理后端。纯张量推理，不处理图像。"""
    def __init__(
        self,
        model_path: str,
        class_list: List[str],
        task_type: Optional[str] = None,   # classify / detect / segment
        half: bool = False,
        device: Optional[str] = None,
        **kwargs
    ): ...

    @abstractmethod
    def __call__(self, input_data: np.ndarray) -> List[np.ndarray]: ...
    def warmup(self): ...
    def get_input_info(self) -> ModelInfo: ...
    def get_output_info(self) -> List[ModelInfo]: ...
    def print_model_info(self): ...
```

| 实现 | 后端 | 备注 |
|------|------|------|
| `OnnxBackend` | ONNX Runtime | `onnxruntime.InferenceSession` |
| `TensorrtBackend` | TensorRT | `trt.Runtime` + CUDA buffer |
| `TritonBackend` | Triton Server | gRPC/HTTP client |

### 2.3 BasePreprocessor / BasePostprocessor

预处理和后处理根据任务类型（classify/detect/segment）有不同的实现：

```python
class BasePreprocessor(ABC):
    """预处理。图像 → 网络输入张量。"""
    @abstractmethod
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray: ...

class BasePostprocessor(ABC):
    """后处理。原始输出 → 结构化结果。"""
    @abstractmethod
    def __call__(self, raw: List[np.ndarray], **kwargs) -> Any: ...
```

### 2.4 BaseDataset

```python
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, idx) -> Tuple[Any, Dict]: ...
    # 返回 (image: np.ndarray(H,W,3), ground_truth: dict)
```

### 2.5 BaseMetrics

```python
class BaseMetrics(ABC):
    """指标计算（有状态累加器）。"""
    @abstractmethod
    def update(self, prediction, ground_truth): ...
    @abstractmethod
    def compute(self) -> Dict[str, float]: ...
    @abstractmethod
    def reset(self): ...
```

### 2.6 BaseEvaluator

```python
class BaseEvaluator(ABC):
    """
    评估器 = Pipeline + Dataset + Metrics.
    负责编排推理循环，metrics 和 visualization 委托 DataFlow-CV.
    """
    def __init__(self, pipeline, dataset, metrics, config): ...
    @abstractmethod
    def run(self) -> Dict[str, float]: ...
    def visualize(self, save_dir, max_samples=100): ...
```

### 2.7 BaseVisualizer

```python
class BaseVisualizer(ABC):
    @abstractmethod
    def draw(self, image, prediction, **kwargs) -> np.ndarray: ...
```

## 3. 推理后端

### 3.1 OnnxBackend

```python
@BACKENDS.register("onnxruntime")
class OnnxBackend(BaseBackend):
    def __init__(self, model_path: str, config: dict = None):
        # config: providers, half, device, ...
    def __call__(self, input_data) -> List[np.ndarray]:
        # ort_session.run(output_names, feed_dict)
```

### 3.2 TensorrtBackend

```python
@BACKENDS.register("tensorrt")
class TensorrtBackend(BaseBackend):
    def __init__(self, engine_path: str, config: dict = None):
        # config: device, max_batch_size, half, ...
    def __call__(self, input_data) -> List[np.ndarray]:
        # H2D → execute_async_v2 → D2H
```

### 3.3 TritonBackend

```python
@BACKENDS.register("triton")
class TritonBackend(BaseBackend):
    def __init__(self, model_name: str, config: dict = None):
        # config: server_url, protocol(grpc/http), ...
    def __call__(self, input_data) -> List[np.ndarray]:
        # grpcclient.InferInput → client.infer → response.as_numpy
```

## 4. 处理器（Processors）

每个视觉任务对应一组 Preprocessor + Postprocessor，每种实现同时提供 NumPy 和 PyTorch 两个版本。

### 4.1 Classify

| 组件 | 实现 | 说明 |
|------|------|------|
| `ClassifyPreprocessor` | npy (PIL resize + crop + normalize) / tch (torchvision) | resize / crop 两种模式 |
| `ClassifyPostprocessor` | npy/tch 共用 softmax | 输出 top-1/top-5 |

### 4.2 Detect

检测任务的预处理和后处理需要处理**YOLO 版本差异**（YOLOv5 vs YOLOv8/YOLO11）。差异主要体现在：
- **输出结构**：v5 维度排列为 `(1, num_dets, 5+nc)`、v8 为 `(1, 84, 8400)` 转置
- **NMS 逻辑**：anchor-based (v5) vs anchor-free (v8)
- **预处理**：letterbox 的 auto pad 参数略有不同

通过 `model_version` 参数区分，避免为每个版本重复实现：

```python
@PROCESSORS.register("detect_preprocess")
class DetectPreprocessor(BasePreprocessor):
    def __init__(self, input_size=640, model_version="v8", half=False):
        # model_version: "v5", "v8", "v11"（v8 和 v11 兼容）
        if model_version in ("v8", "v11"):
            self.auto_pad = True
        else:  # v5
            self.auto_pad = False
    def __call__(self, image, **kwargs) -> np.ndarray: ...
```

| 组件 | 实现 | 说明 |
|------|------|------|
| `DetectPreprocessor` | npy (OpenCV letterbox) / tch (letterbox) | stride 对齐、pad，支持 model_version 参数 |
| `DetectPostprocessor` | npy NMS / tch NMS | NMS + scale_boxes + clip_boxes，适配 v5/v8 输出格式差异 |
| `detect/ops.py` | 共享算子 | xywh2xyxy, scale_boxes, clip_boxes, nms 等 |

### 4.3 Segment（实例分割）

| 组件 | 实现 | 说明 |
|------|------|------|
| `SegmentPreprocessor` | 同 Detect | letterbox |
| `SegmentPostprocessor` | npy / tch | NMS + process_mask + crop_mask + scale_image |

### 4.4 Semantic Segmentation

| 组件 | 实现 | 说明 |
|------|------|------|
| `SemanticSegPreprocessor` | resize + normalize | ImageNet 或数据集统计 |
| `SemanticSegPostprocessor` | argmax + colormap | HWC uint8 掩码 |

### 4.5 Multi-modal

| 组件 | 实现 | 说明 |
|------|------|------|
| `ImagePreprocessor` | CLIP 标准预处理 | resize 224 + center crop + normalize |
| `TextPreprocessor` | CLIP tokenizer | 文本 → token IDs |
| `Postprocessor` | softmax + ranking | similarity → probability |

## 5. 评估与 DataFlow-CV 桥接

### 5.1 评估分层架构

`modelflow/evaluators/` 是评估**框架层**，提供统一接口编排推理循环；
`eval/` 目录下的脚本是 **benchmark 入口**，调用 evaluators 完成特定后端的评估。

```
eval/runtime/bench_yolov8_onnx_npy.py  ← 入口脚本（可独立运行）
    └── 调用 → modelflow/evaluators/detect.py  ← 评估编排
                    ├── Pipeline(DetectPreprocessor + OnnxBackend + DetectPostprocessor)
                    ├── Dataset(COCO)
                    └── DataFlow-CV Evaluator(mAP)
```

**职责划分：**

| 层 | 目录 | 职责 | 是否可独立运行 |
|----|------|------|:-------------:|
| 入口 | `eval/<backend>/bench_*.py` | 组装参数、启动评估 | ✅ |
| 框架 | `modelflow/evaluators/` | 编排推理循环、收集结果、调用 metric | ❌ |
| 指标 | `modelflow/metrics/` 或 DataFlow-CV | 具体指标计算 | ❌ |

### 5.2 评估桥接

Detection 和 Segmentation 评估**委托 DataFlow-CV 实现**：

```python
# modelflow/evaluators/detect.py
from dataflow.evaluate import DetectionEvaluator as DFDetectionEvaluator

class DetectEvaluator(BaseEvaluator):
    def run(self) -> Dict[str, float]:
        # 1. Pipeline 遍历数据集 → 收集 COCO 格式预测
        coco_predictions = self._run_inference()

        # 2. 保存预测 JSON
        pred_json = self._save_predictions(coco_predictions)

        # 3. 委托 DataFlow-CV 计算 mAP
        df_eval = DFDetectionEvaluator(verbose=...)
        result = df_eval.evaluate(self.dataset.gt_json, pred_json)

        # 4. 返回标准 metrics dict
        return self._to_metrics(result)
```

### 5.3 可视化桥接

Detection 和 Segmentation 可视化**委托 DataFlow-CV 实现**：

```python
# modelflow/viz/detect.py
from dataflow.visualize import COCOVisualizer as DFCOCOVisualizer

class DetectVisualizer(BaseVisualizer):
    def draw(self, image, prediction, **kwargs):
        visualizer = DFCOCOVisualizer(
            annotation_file=pred_json,
            image_dir=self.image_dir,
            is_save=True,
            output_dir=self.output_dir,
        )
        result = visualizer.visualize()
        return result.data
```

### 5.4 本地实现

Classification 和 Semantic Segmentation 的 metrics 在 `modelflow/metrics/` 本地实现（DataFlow-CV 暂无）：

```python
# modelflow/metrics/classification.py
class ClassificationMetrics(BaseMetrics):
    """混淆矩阵 → Accuracy / Precision / Recall / F1"""
    def __init__(self, num_classes): ...
    def update(self, pred_class, gt_class): ...   # 累加混淆矩阵
    def compute(self) -> dict: ...
    def reset(self): ...
```

## 6. 注册机制

```python
# modelflow/core/registry.py
class Registry:
    """组件注册器。支持装饰器注册和按名构建。"""
    def __init__(self, name): ...
    def register(self, name=None) -> Callable: ...  # 装饰器
    def get(self, name) -> type: ...
    def build(self, name, **kwargs) -> Any: ...
    def list(self) -> List[str]: ...

# 全局注册器
BACKENDS = Registry("backends")
PROCESSORS = Registry("processors")
DATASETS = Registry("datasets")
METRICS = Registry("metrics")
EVALUATORS = Registry("evaluators")
```

### 6.1 扩展示例：新增后端

```python
@BACKENDS.register("openvino")
class OpenVINOBackend(BaseBackend):
    def __call__(self, input_data):
        ...
```

### 6.2 扩展示例：新增视觉任务

```python
# 1. TaskType 添加
class TaskType(Enum):
    ...
    POSE = "pose"

# 2. 处理器
@PROCESSORS.register("pose_preprocess")
class PosePreprocessor(BasePreprocessor): ...

@PROCESSORS.register("pose_postprocess")
class PosePostprocessor(BasePostprocessor): ...

# 3. 评估
@METRICS.register("pose")
class PoseMetrics(BaseMetrics): ...

@EVALUATORS.register("pose")
class PoseEvaluator(BaseEvaluator): ...
```

## 7. Pipeline 工厂

```python
# modelflow/pipelines/detect.py
def create_detect_pipeline(
    backend: str = "onnxruntime",    # onnxruntime / tensorrt / triton
    processor: str = "numpy",         # numpy / torch
    model_path: str,
    class_list: List[str],
    input_size: int = 640,
    **kwargs
) -> InferencePipeline:
    """创建检测 Pipeline 的便捷工厂函数。"""
    ...

# 使用
pipeline = create_detect_pipeline(
    backend="onnxruntime",
    processor="numpy",
    model_path="yolov8s.onnx",
    class_list=["person", "car"],
)
result = pipeline(image, conf_thres=0.25, iou_thres=0.45)
# result = { "boxes": ndarray(N,4), "scores": ndarray(N,), "class_ids": ndarray(N,) }
```

## 8. 枚举类型

```python
# modelflow/core/types.py
class TaskType(str, Enum):
    CLASSIFY = "classify"
    DETECT = "detect"
    INSTANCE_SEGMENT = "instance_segment"
    SEMANTIC_SEGMENT = "semantic_segment"
    MULTIMODAL = "multimodal"

class BackendType(str, Enum):
    ONNXRUNTIME = "onnxruntime"
    TENSORRT = "tensorrt"
    TRITON = "triton"

class ProcessorType(str, Enum):
    NUMPY = "numpy"
    TORCH = "torch"

@dataclass
class ModelInfo:
    name: str
    shape: List[int]
    dtype: np.dtype
```
