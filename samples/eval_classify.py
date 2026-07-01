# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : eval_classify.py
@Author  : zj
@Description: 分类模型评估基准 — ImageNet / CIFAR / 自定义分类数据集

用法:
    python3 samples/eval_classify.py \
        --model models/runtime/efficientnet_b0.onnx \
        --data /path/to/imagenet

    python3 samples/eval_classify.py \
        --model models/runtime/resnet50.onnx \
        --data /path/to/cifar10 --dataset cifar10 --input-size 32

模型元数据和延迟测量请使用: python3 samples/parse_model.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modelflow.pipelines import create_classify_pipeline
from eval import ClassifyEvaluator
from data import build_dataset, get_class_names
from utils import Profile


def parse_opt():
    parser = argparse.ArgumentParser(description="ModelFlow 分类评估")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 Triton 模型名")
    parser.add_argument("--data", type=str, required=True,
                        help="数据集根目录（如 /path/to/imagenet，包含 val/ 子目录）")
    parser.add_argument("--dataset", type=str, default="imagenet",
                        help="数据集配置名: imagenet / cifar10 / cifar100")
    parser.add_argument("--backend", type=str, default="onnxruntime",
                        choices=["onnxruntime", "tensorrt", "triton"],
                        help="推理后端")
    parser.add_argument("--input-size", type=int, default=224,
                        help="输入尺寸 (ImageNet=224, CIFAR=32)")
    return parser.parse_args()


def main():
    args = parse_opt()

    class_list = get_class_names(args.dataset)

    with Profile("pipeline.build"):
        pipeline = create_classify_pipeline(
            model_path=args.model,
            class_list=class_list,
            backend=args.backend,
            input_size=args.input_size,
        )

    dataset = build_dataset(args.dataset, path=args.data, val="val")

    evaluator = ClassifyEvaluator(pipeline, dataset)

    print(f"🔧 Task: classify, Backend: {args.backend}")
    print(f"📂 Data: {args.dataset} ({args.data}), Input size: {args.input_size}")

    with Profile("eval") as perf:
        results = evaluator.run()

    print(f"⏱ Total time: {perf.elapsed:.2f}s")
    print(f"📊 Results: {results}")


if __name__ == "__main__":
    main()
