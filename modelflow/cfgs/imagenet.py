# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : imagenet.py
@Author  : zj
@Description: ImageNet 数据集类别定义（1000 类）
"""

# ImageNet 1000 类 ID → 名称映射
# 这里列出标准 1000 类的前 10 个和最后 10 个作为示意
# 完整列表可在代码注释中引用，实际使用时通过 torchvision 的类别列表
class_list = ["n%08d" % i for i in range(1000)]  # Placeholder

# 实际使用时，可通过 torchvision 或从文件加载完整映射
def get_imagenet_classes() -> list:
    """获取 ImageNet 1000 类名称列表

    生产环境下建议从文件加载完整映射：
        from torchvision.models import ResNet18_Weights
        categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    """
    try:
        from torchvision.models import ResNet18_Weights
        categories = ResNet18_Weights.IMAGENET1K_V1.meta.get("categories")
        if categories:
            return categories
    except (ImportError, AttributeError):
        pass
    return class_list
