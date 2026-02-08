# README

## Install

```angular2html
# Replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU.
$ pip install open_clip_torch
```

## OpenCLIP (ViT-B-32, laion400m_e32) 在 CIFAR-10 和 CIFAR-100 上的 Zero-Shot 和 Linear Probe 评估结果

| 方法                    | 数据集      | 准确率 (Accuracy)  | 提升（对比 Zero-Shot 单模板） |
|-----------------------|----------|-----------------|----------------------|
| **Zero-Shot（单模板）**    | CIFAR-10 | 88.66%          | -                    |
| **Zero-Shot（20模板集成）** | CIFAR-10 | 88.88% (+0.22%) | +0.22%               |
| **Linear Probe**      | CIFAR-10 | **94.75%**      | **+6.09%**           |

| 方法                    | 数据集       | 准确率 (Accuracy)  | 提升（对比 Zero-Shot 单模板） |
|-----------------------|-----------|-----------------|----------------------|
| **Zero-Shot（单模板）**    | CIFAR-100 | 67.02%          | -                    |
| **Zero-Shot（20模板集成）** | CIFAR-100 | 67.90% (+0.88%) | +0.88%               |
| **Linear Probe**      | CIFAR-100 | **78.79%**      | **+11.77%**          |