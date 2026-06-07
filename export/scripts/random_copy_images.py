# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : random_copy_images.py
@Author  : zj
@Description: 从源目录递归扫描图片，随机抽取 N 张复制到目标目录

用于从 COCO/ImageNet 数据集中随机抽样子集，作为 INT8 校准数据源。

用法：
    python3 export/scripts/random_copy_images.py \\
        /path/to/source /path/to/target 100
"""

import os
import shutil
import random
import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def get_unique_filename(target_path: Path, filename: str) -> str:
    """如果文件名冲突，添加后缀 _1, _2 等"""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while (target_path / new_filename).exists():
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename


def random_copy_images(source: str, target: str, n: int = 100):
    """从 source 递归收集图片，随机选 n 张复制到 target

    Args:
        source: 源目录
        target: 目标目录
        n: 抽取数量
    """
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()

    if not source_path.exists():
        print(f"❌ Source not found: {source_path}")
        return

    target_path.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Scanning {source_path} ...")

    # 递归收集所有图片
    all_images = []
    for root, _dirs, files in os.walk(source_path):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                all_images.append(os.path.join(root, f))

    total = len(all_images)
    print(f"✅ Found {total} images")

    if total == 0:
        return

    n = min(n, total)
    selected = random.sample(all_images, n)
    print(f"🎲 Randomly selected {len(selected)} images, copying...")

    success = 0
    for src_file in selected:
        filename = os.path.basename(src_file)
        unique_name = get_unique_filename(target_path, filename)
        dst_file = target_path / unique_name
        try:
            shutil.copy2(src_file, dst_file)
            success += 1
        except Exception as e:
            print(f"  ❌ {filename}: {e}")

    print("-" * 30)
    print(f"🎉 Done! Copied {success} images to {target_path}")


def parse_opt():
    parser = argparse.ArgumentParser(description="随机抽取图片")
    parser.add_argument("source", type=str, help="源目录")
    parser.add_argument("target", type=str, help="目标目录")
    parser.add_argument("count", type=int, nargs="?", default=100, help="抽取数量")
    return parser.parse_args()


def main():
    args = parse_opt()
    random_copy_images(args.source, args.target, args.count)


if __name__ == "__main__":
    main()
