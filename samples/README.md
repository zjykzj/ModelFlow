# Samples — ModelFlow 使用入口

## 推理（单图 + 可视化）

```bash
# 检测
python3 samples/infer.py --task detect --model models/runtime/yolov8s.onnx --image assets/bus.jpg

# 分类
python3 samples/infer.py --task classify --model models/runtime/efficientnet_b0.onnx --image assets/cat.jpg --classes imagenet --input-size 224

# 分割
python3 samples/infer.py --task segment --model models/runtime/yolov8s-seg.onnx --image assets/bus.jpg

# 语义分割
python3 samples/infer.py --task semantic_seg --model models/runtime/segformer.onnx --image assets/bus.jpg

# GPU 后端
python3 samples/infer.py --task detect --model models/tensorrt/yolov8s_fp16.engine --backend tensorrt --image assets/bus.jpg

# 保存 + 显示
python3 samples/infer.py --task detect --model models/runtime/yolov8s.onnx --image assets/bus.jpg --save result.jpg --show
```

## 评估（批量数据集）

分类、检测、分割各有独立脚本，参数按任务定制：

```bash
# 分类评估 (ImageNet)
python3 samples/eval_classify.py --model models/runtime/efficientnet_b0.onnx --data /path/to/imagenet

# 检测评估 (COCO mAP)
python3 samples/eval_detect.py --model models/runtime/yolov8s.onnx --data /path/to/coco
python3 samples/eval_detect.py --model models/runtime/yolov5s.onnx --data /path/to/coco --model-version v5
python3 samples/eval_detect.py --model models/tensorrt/yolov8s_fp16.engine --backend tensorrt --data /path/to/coco --save-pred results.json

# 分割评估 (COCO segm mAP)
python3 samples/eval_segment.py --model models/runtime/yolov8s-seg.onnx --data /path/to/coco
```

## 模型分析（三阶段）

模型选型和部署决策需要从三个维度独立评估，各自使用专用脚本：

```
第一阶段                       第二阶段                    第三阶段
analyze_model.py             bench_model.py             eval_*.py
(架构分析)                    (延迟基准)                  (精度验证)
                                                         
.onnx ──→ 参数量              .onnx ──→ 推理延迟         .onnx ──→ mAP
          FLOPs               .engine ─→ 推理延迟        .engine ─→ mAP (精度损失?)
          I/O shape           Triton ──→ 推理延迟        Triton ──→ mAP
          推断 task/ver/nc    管线各阶段耗时               per-class 指标
                                                         
仅支持 ONNX                  ONNX / TRT / Triton        ONNX / TRT / Triton
不需要 GPU                   视后端而定                  视后端而定 + 数据集
毫秒级                       秒级                        分钟~小时级
一次性（架构常数）            多次（调优对比）            多次（版本验证）
```

### 第一阶段：架构分析 (`analyze_model.py`)

获取模型的**理论常数**——参数量和 FLOPs 是模型架构的固有属性，不会因为格式转换（PT→ONNX→TRT）而改变，只需在 ONNX 上测一次。

同时自动推断推理配置：任务类型、YOLO 版本、类别数、输入尺寸。

```bash
# 检测模型
python3 samples/analyze_model.py --model models/yolov8s.onnx

# 分割模型
python3 samples/analyze_model.py --model models/yolov8s-seg.onnx

# 分类模型
python3 samples/analyze_model.py --model models/efficientnet_b0.onnx

# 保存为 JSON 供后续脚本消费
python3 samples/analyze_model.py --model models/yolov8s.onnx --json meta.json
```

### 第二阶段：延迟基准 (`bench_model.py`)

对比**不同后端**（ONNX Runtime / TensorRT / Triton）的推理速度。这是硬件+运行时层面的指标，同一架构在不同后端上延迟差异显著。

两种测量模式：
- `backend`：纯推理延迟（dummy tensor，跳过预处理/后处理），用于对比不同后端的纯推理吞吐
- `pipeline`：完整管线延迟（预处理 + 推理 + 后处理 三段计时），模拟真实场景端到端耗时

```bash
# 后端推理延迟（对比 ONNX vs TRT 纯推理速度）
python3 samples/bench_model.py --model models/yolov8s.onnx --task detect
python3 samples/bench_model.py --model models/yolov8s_fp16.engine --backend tensorrt --task detect

# 完整管线延迟（真实图，含 pre/infer/post 分段耗时）
python3 samples/bench_model.py --model models/yolov8s.onnx --task detect --mode pipeline --image assets/bus.jpg

# 自定义迭代次数
python3 samples/bench_model.py --model models/yolov8s.onnx --task detect --iters 500 --warmup 50

# Triton 后端
python3 samples/bench_model.py --model Detect_COCO_YOLOv8s_TRT --backend triton --task detect

# 导出 JSON
python3 samples/bench_model.py --model models/yolov8s.onnx --task detect --json latency.json
```

### 第三阶段：精度验证 (`eval_*.py`)

在目标数据集上验证模型精度。**TensorRT 量化（FP16/INT8）后必须测精度**——加速倍数再高，如果 mAP 显著下降也不值得。

```bash
# ONNX 基线
python3 samples/eval_detect.py --model models/yolov8s.onnx --data /path/to/coco

# TensorRT FP16 —— 验证精度损失是否在可接受范围（通常 < 0.5 mAP）
python3 samples/eval_detect.py --model models/yolov8s_fp16.engine --backend tensorrt --data /path/to/coco

# TensorRT INT8 —— 验证量化精度损失（通常 < 1.0 mAP）
python3 samples/eval_detect.py --model models/yolov8s_int8.engine --backend tensorrt --data /path/to/coco
```

### 为什么参数量/FLOPs 只测 ONNX？

| 指标 | ONNX | TensorRT Engine |
|------|------|----------------|
| 参数量 | ✅ 从 `graph.initializer` 累加 | ❌ 引擎是不透明二进制，无法遍历权重 |
| FLOPs | ✅ 遍历计算图 + shape_inference 估算 | ❌ TRT 会做 kernel fusion，原始 FLOPs 已不适用 |

参数量和 FLOPs 是**模型架构的出生证明**，测一次就够了。延迟和精度才是**模型在不同后端上的工作表现**，每个后端都要分别测。

## 文件说明

| 文件 | 类型 | 用途 |
|------|------|------|
| `analyze_model.py` | 架构分析 | ONNX 参数量/FLOPs/I/O shape + 推断 task/version/nc |
| `bench_model.py` | 延迟基准 | 后端推理/管线延迟，支持 ONNX/TRT/Triton |
| `eval_classify.py` | 精度验证 | 分类评估 — ImageNet Accuracy / Precision / Recall / F1 |
| `eval_detect.py` | 精度验证 | 检测评估 — COCO mAP (DataFlow-CV 桥接) |
| `eval_segment.py` | 精度验证 | 分割评估 — COCO segm mAP (DataFlow-CV 桥接) |
| `infer.py` | 推理 | 统一推理入口 — 单图 + 可视化，支持 `--save`/`--show` |
| `utils.py` | 工具 | 可视化 — `draw_detections` / `draw_segmentation` / `draw_classification` |
| `parse_model.py` | 废弃 | ⚠️ 已拆分为 `analyze_model.py` + `bench_model.py` |
