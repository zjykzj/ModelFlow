# ModelFlow Python Inference Package

Python 推理核心包，提供 **Pipeline = Preprocessor + Backend + Postprocessor** 统一架构。

## 快速开始

### 分类推理

```python
from modelflow.pipelines import create_classify_pipeline

pipeline = create_classify_pipeline(
    model_path="efficientnet_b0.onnx",
    class_list=["cat", "dog", "bird"],
    backend="onnxruntime",
)

result = pipeline(image)
print(result["class_ids"], result["scores"])  # top-5
```

### 检测推理

```python
from modelflow.pipelines import create_detect_pipeline
from modelflow.config import COCO_CLASSES

pipeline = create_detect_pipeline(
    model_path="yolov8s.onnx",
    class_list=COCO_CLASSES,
    backend="onnxruntime",
)

result = pipeline(image, conf_thres=0.25, iou_thres=0.45)
# result = {"boxes": ndarray(N,4), "scores": ndarray(N,), "class_ids": ndarray(N,)}
```

### 实例分割推理

```python
from modelflow.pipelines import create_segment_pipeline
from modelflow.config import COCO_CLASSES

pipeline = create_segment_pipeline(
    model_path="yolov8s-seg.onnx",
    class_list=COCO_CLASSES,
)

result = pipeline(image)
# result = {"boxes": ..., "scores": ..., "class_ids": ..., "masks": ...}
```

### 语义分割推理

```python
from modelflow.pipelines import create_semantic_seg_pipeline

pipeline = create_semantic_seg_pipeline(
    model_path="segformer.onnx",
)

result = pipeline(image)
# result = {"class_map": ndarray(H,W), "colormap": ndarray(H,W,3)}
```

## Pipeline = Preprocessor + Backend + Postprocessor

```
image → Preprocessor → tensor → Backend → raw → Postprocessor → result
```

| 组件 | 接口 | 实现 |
|------|------|------|
| Preprocessor | `BasePreprocessor` | classify / detect / segment / semantic_seg |
| Backend | `BaseBackend` | OnnxBackend / TensorrtBackend / TritonBackend |
| Postprocessor | `BasePostprocessor` | 按任务：softmax top-k、NMS、proto mask、argmax |

## 支持的任务

| 任务 | Preprocessor | Backend | Postprocessor |
|------|-------------|---------|---------------|
| Classification | Resize+Crop+Normalize | ONNX/TRT/Triton | softmax top-k |
| Detection | LetterBox → /255 | ONNX/TRT/Triton | NMS + scale |
| Instance Seg | LetterBox → /255 | ONNX/TRT/Triton | NMS + proto mask |
| Semantic Seg | Resize+Normalize | ONNX/TRT/Triton | argmax + colormap |

## Backend 扩展

新增 Backend 通过直接构造，无需注册机制：

```python
from modelflow.interfaces import BaseBackend

class MyBackend(BaseBackend):
    def __call__(self, tensor):
        # tensor: np.ndarray (N,C,H,W) float32
        # return: List[np.ndarray]
        ...

# 在对应 pipeline 工厂中添加入口
```

## 模块结构

```
modelflow/
├── __init__.py       # 版本 + 公共接口 re-export
├── interfaces.py     # ABC: InferencePipeline, BaseBackend, BasePreprocessor, BasePostprocessor
├── types.py          # ModelInfo dataclass
├── config.py         # ModelConfig + COCO_CLASSES (80), IMAGENET_CLASSES (1000)
├── backends/         # ONNX / TensorRT / Triton 推理后端（惰性导入）
├── processors/       # 按任务组织的 pre/post 处理器
│   ├── classify/     #   Resize+Crop+Normalize → softmax top-k
│   ├── detect/       #   LetterBox → NMS+box decode (YOLOv5/v8/v11)
│   ├── segment/      #   LetterBox → NMS+proto mask decode
│   └── semantic_seg/ #   Resize+Normalize → argmax+colormap
└── pipelines/        # Pipeline 工厂函数（_ensure_backend + _build_backend）
    ├── classify.py
    ├── detect.py
    ├── segment.py
    └── semantic_seg.py
```
