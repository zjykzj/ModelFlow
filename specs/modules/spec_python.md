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
├── backends/               # 推理后端
│   ├── base.py             # BaseBackend
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
├── evaluators/             # 评估器
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

    用法:
        pipeline = InferencePipeline(preprocessor, backend, postprocessor)
        result = pipeline(image, conf_thres=0.25, iou_thres=0.45)  # 端到端
        raw = pipeline.infer(tensor)  # 仅推理（评估用）
    """
    def __init__(self, preprocessor, backend, postprocessor): ...
    def __call__(self, image, **kwargs) -> Any: ...
    def infer(self, tensor: np.ndarray) -> List[np.ndarray]: ...
    def warmup(self): ...
```

### 2.2 BaseBackend

```python
class BaseBackend(ABC):
    """推理后端。纯张量推理，不处理图像。"""
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

| 组件 | 实现 | 说明 |
|------|------|------|
| `DetectPreprocessor` | npy (OpenCV letterbox) / tch (letterbox) | stride 对齐、pad |
| `DetectPostprocessor` | npy NMS / tch NMS | NMS + scale_boxes + clip_boxes |
| `detect/ops.py` | 共享算子 | xywh2xyxy, scale_boxes, clip_boxes |

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

### 5.1 评估桥接

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

### 5.2 可视化桥接

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

### 5.3 本地实现

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
