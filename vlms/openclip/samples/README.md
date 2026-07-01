# README

## Install

```angular2html
# Replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU.
$ pip install open_clip_torch
```

## OpenCLIP (ViT-B-32, laion400m_e32) 在 CIFAR-10 和 CIFAR-100 上的完整评估结果

| 方法 / 分类头                     | 数据集      | 准确率 (Accuracy) | 提升（对比 Zero-Shot 单模板） |
|------------------------------|----------|----------------|----------------------|
| **Zero-Shot（单模板）**           | CIFAR-10 | 88.66%         | -                    |
| **Zero-Shot（20模板集成）**        | CIFAR-10 | 88.88%         | +0.22%               |
| **Linear Probe + LR**        | CIFAR-10 | **94.85%**     | **+6.19%**           |
| **Linear Probe + KNN (k=1)** | CIFAR-10 | 91.91%         | +3.25%               |
| **Linear Probe + KNN (k=5)** | CIFAR-10 | 93.56%         | +4.90%               |

| 方法 / 分类头                     | 数据集       | 准确率 (Accuracy) | 提升（对比 Zero-Shot 单模板） |
|------------------------------|-----------|----------------|----------------------|
| **Zero-Shot（单模板）**           | CIFAR-100 | 67.02%         | -                    |
| **Zero-Shot（20模板集成）**        | CIFAR-100 | 67.90%         | +0.88%               |
| **Linear Probe + LR**        | CIFAR-100 | **78.70%**     | **+11.68%**          |
| **Linear Probe + KNN (k=1)** | CIFAR-100 | 70.83%         | +3.81%               |
| **Linear Probe + KNN (k=5)** | CIFAR-100 | 73.32%         | +6.30%               |