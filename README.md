<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

 <div align="center"><a title="" href="git@github.com:zjykzj/ModelFlow.git"><img align="center" src="./assets/logos/ModelFlow.svg"></a></div>

<p align="center">
  Model Eval & Export & Infer
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

<!-- 本仓库的目的是为了更好的部署计算机视觉算法，特别是目标分类、目标检测以及实例分割算法的实现。

最开始我想要设计统一的架构，通过模块化范式来适配不同的前后处理、不同的网络模块实现以及不同的推理引擎，类似于常用的热门仓库。但是这种方式很难推进下去，每次想要加入新的算法，我需要经常反复的思考如何将该算法按照目前架构进行拆分，如何适配每个模块的输入输出。这些问题让我心力憔悴，有一段时间甚至对仓库优化都丧失了热情。

我思考了很久，确认我应该是陷入了某种开发困境，为了追求设计上的完美无限增大了工程开发的复杂度，在意识到过度设计的问题后，我打算重新开始。在新的开发中，我会尽可能的聚焦于这个仓库的目标：模型转换以及模型推理，尽量减少架构设计的内容。把主要精力集中在算法部署上。

注：之前的实现备份在[v0.1.0](https://github.com/zjykzj/ModelFlow/tree/v0.1.0)。 -->

The purpose of this repository is to better deploy computer vision algorithms, especially the implementation of object
classification, object detection, and instance segmentation algorithms.

At first, I wanted to design a unified architecture that would adapt to different pre-processing and post-processing,
network module implementations, and inference engines through a modular paradigm, similar to commonly used popular
repositories. But this approach is difficult to push forward. Every time I want to add a new algorithm, I need to
repeatedly think about how to split the algorithm according to the current architecture and how to adapt the input and
output of each module. These issues have left me exhausted and even lost my passion for warehouse optimization for a
period of time.

I have thought for a long time and confirmed that I may have fallen into some kind of development dilemma. In order to
pursue design perfection, the complexity of engineering development has been infinitely increased. After realizing the
problem of Over-Engineering, I plan to start over. In the new development, I will focus as much as possible on the goals
of this repository: model transformation and model inference, and minimize the content of architecture design. Focus the
main energy on algorithm deployment.

Note: The previous implementation was in [v0.1.0](https://github.com/zjykzj/ModelFlow/tree/v0.1.0).

## YOLOv5/YOLOv8 Eval

| Model      | Inference Backend | Pre/Post-processing | Implementation File              | Implemented |
|------------|-------------------|---------------------|----------------------------------|-------------|
| YOLOv5     | ONNX Runtime      | PyTorch             | `yolov5_runtime_w_torch.py`      | ✅           |
| YOLOv5     | ONNX Runtime      | NumPy               | `yolov5_runtime_w_numpy.py`      | ✅           |
| YOLOv5     | TensorRT          | NumPy               | `yolov5_tensorrt_w_numpy.py`     | ✅           |
| YOLOv5     | Triton Server     | NumPy               | `yolov5_triton_w_numpy.py`       | ✅           |
| YOLOv8     | ONNX Runtime      | PyTorch             | `yolov8_runtime_w_torch.py`      | ✅           |
| YOLOv8     | ONNX Runtime      | NumPy               | `yolov8_runtime_w_numpy.py`      | ✅           |
| YOLOv8     | TensorRT          | NumPy               | `yolov8_tensorrt_w_numpy.py`     | ✅           |
| YOLOv8     | Triton Server     | NumPy               | `yolov8_triton_w_numpy.py`       | ✅           |
| YOLOv8-seg | ONNX Runtime      | PyTorch             | `yolov8_seg_runtime_w_torch.py`  | ✅           |
| YOLOv8-seg | ONNX Runtime      | NumPy               | `yolov8_seg_runtime_w_numpy.py`  | ✅           |
| YOLOv8-seg | TensorRT          | NumPy               | `yolov8_seg_tensorrt_w_numpy.py` | ✅           |
| YOLOv8-seg | Triton Server     | NumPy               | `yolov8_seg_triton_w_numpy.py`   | ✅           |

## CLIP/OpenCLIP Eval

CLIP 和 OpenCLIP 模型在 CIFAR-10 和 CIFAR-100 上的 Zero-Shot 和 Linear Probe 评估结果

### CIFAR-10 数据集

| 方法 / 分类头                     | 模型       | 准确率 (Accuracy) | 提升（对比 Zero-Shot 单模板） |
|------------------------------|----------|----------------|----------------------|
| **Zero-Shot（单模板）**           | CLIP     | 88.80%         | —                    |
| **Zero-Shot（20模板集成）**        | CLIP     | 89.52%         | +0.72%               |
| **Linear Probe + LR**        | CLIP     | 94.32%         | +5.52%               |
| **Zero-Shot（单模板）**           | OpenCLIP | 88.66%         | —                    |
| **Zero-Shot（20模板集成）**        | OpenCLIP | 88.88%         | +0.22%               |
| **Linear Probe + LR**        | OpenCLIP | **94.85%**     | **+6.19%**           |
| **Linear Probe + KNN (k=1)** | OpenCLIP | 91.91%         | +3.25%               |
| **Linear Probe + KNN (k=5)** | OpenCLIP | 93.56%         | +4.90%               |

### CIFAR-100 数据集

| 方法 / 分类头                     | 模型       | 准确率 (Accuracy) | 提升（对比 Zero-Shot 单模板） |
|------------------------------|----------|----------------|----------------------|
| **Zero-Shot（单模板）**           | CLIP     | 61.70%         | —                    |
| **Zero-Shot（20模板集成）**        | CLIP     | 63.96%         | +2.26%               |
| **Linear Probe + LR**        | CLIP     | 75.62%         | +13.92%              |
| **Zero-Shot（单模板）**           | OpenCLIP | 67.02%         | —                    |
| **Zero-Shot（20模板集成）**        | OpenCLIP | 67.90%         | +0.88%               |
| **Linear Probe + LR**        | OpenCLIP | **78.70%**     | **+11.68%**          |
| **Linear Probe + KNN (k=1)** | OpenCLIP | 70.83%         | +3.81%               |
| **Linear Probe + KNN (k=5)** | OpenCLIP | 73.32%         | +6.30%               |

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/ModelFlow/issues) or submit PRs.

## License

[Apache License 2.0](LICENSE) © 2021 zjykzj