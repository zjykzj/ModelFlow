# README

## Install

```angular2html
# Replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU.
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

## CLIP 模型在 CIFAR-10 和 CIFAR-100 上的 Zero-Shot 和 Linear Probe 评估结果

| 方法                    | 数据集      | 准确率 (Accuracy)  | 提升（对比 Zero-Shot 单模板） |
|-----------------------|----------|-----------------|----------------------|
| **Zero-Shot（单模板）**    | CIFAR-10 | 88.80%          | -                    |
| **Zero-Shot（20模板集成）** | CIFAR-10 | 89.52% (+0.72%) | +0.72%               |
| **Linear Probe**      | CIFAR-10 | **94.32%**      | **+5.52%**           |

| 方法                    | 数据集       | 准确率 (Accuracy)  | 提升（对比 Zero-Shot 单模板） |
|-----------------------|-----------|-----------------|----------------------|
| **Zero-Shot（单模板）**    | CIFAR-100 | 61.70%          | -                    |
| **Zero-Shot（20模板集成）** | CIFAR-100 | 63.96% (+2.26%) | +2.26%               |
| **Linear Probe**      | CIFAR-100 | **75.62%**      | **+13.92%**          |