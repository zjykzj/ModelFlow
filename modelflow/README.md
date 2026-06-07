# ModelFlow Python Inference Package

Python 推理/评估/可视化核心包，提供 **Pipeline = Preprocessor + Backend + Postprocessor** 统一架构。

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
from modelflow.cfgs.coco import class_list

pipeline = create_detect_pipeline(
    model_path="yolov8s.onnx",
    class_list=class_list,
    backend="onnxruntime",
)

result = pipeline(image, conf_thres=0.25, iou_thres=0.45)
# result = {"boxes": ndarray(N,4), "scores": ndarray(N,), "class_ids": ndarray(N,)}
```

### 实例分割推理

```python
from modelflow.pipelines import create_segment_pipeline
from modelflow.cfgs.coco import class_list

pipeline = create_segment_pipeline(
    model_path="yolov8s-seg.onnx",
    class_list=class_list,
)

result = pipeline(image)
# result = {"boxes": ..., "scores": ..., "class_ids": ..., "masks": ...}
```

### 评估

```python
from modelflow.evaluators import DetectEvaluator
from modelflow.datasets import COCODetectionDataset
from modelflow.cfgs.coco import class_list

dataset = COCODetectionDataset("val2017/", class_list, anno_json="annotations.json")
evaluator = DetectEvaluator(pipeline, dataset, gt_json="annotations.json")
metrics = evaluator.run(save_pred_json="results.json")
# {"mAP": 0.5, "AP50": 0.7, ...}
```

## Pipeline = Preprocessor + Backend + Postprocessor

```
image → Preprocessor → tensor → Backend → raw → Postprocessor → result
                                                                      ↓
                                                              Evaluator 编排
```

| 组件 | 接口 | 实现 |
|------|------|------|
| Preprocessor | `BasePreprocessor` | classify / detect / segment / semantic_seg / multimodal |
| Backend | `BaseBackend` | OnnxBackend / TensorrtBackend / TritonBackend |
| Postprocessor | `BasePostprocessor` | 按任务：NMS、softmax、argmax、相似度 |
| Dataset | `BaseDataset` | COCO detection/segment、分类目录 |
| Evaluator | `BaseEvaluator` | Detect / Classify / Segment (DataFlow-CV 桥接) |

## 支持的任务和模型

| 任务 | Preprocessor | Backend | Postprocessor |
|------|-------------|---------|---------------|
| Classification | Resize+Crop+Normalize | ONNX/TRT/Triton | softmax top-k |
| Detection | LetterBox → /255 | ONNX/TRT/Triton | NMS + scale |
| Instance Seg | LetterBox → /255 | ONNX/TRT/Triton | NMS + proto mask |
| Semantic Seg | Resize+Normalize | ONNX/TRT/Triton | argmax + colormap |
| CLIP | CLIP standard | ONNX/TRT/Triton | similarity ranking |

## 注册机制

新增后端、任务、数据集无需改框架代码：

```python
from modelflow.core import BACKENDS, BaseBackend

@BACKENDS.register("my_backend")
class MyBackend(BaseBackend):
    def __call__(self, input_data):
        # your logic
        pass
```

## 模块结构

```
modelflow/
├── core/        抽象基类、注册机制、类型枚举、配置
├── cfgs/        COCO 80 类、ImageNet 1000 类
├── backends/    ONNX / TensorRT / Triton 推理后端
├── processors/  按任务组织的前/后处理器
├── pipelines/   Pipeline 工厂函数
├── datasets/    数据集加载器
├── evaluators/  评估编排器
├── metrics/     本地指标实现
├── viz/         可视化（OpenCV + DataFlow-CV 桥接）
└── utils/       日志、计时器
```
