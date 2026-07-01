# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : random_copy_images.py
@Author  : zj
@Description: Recursively scan images under a source directory, randomly pick N images and copy them to a target directory.

Used to randomly sample subsets from COCO/ImageNet datasets as INT8 calibration data sources.

Usage:
    python3 export/tensorrt/scripts/random_copy_images.py \\
        /path/to/source /path/to/target 100
"""

import os
import shutil
import random
import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def get_unique_filename(target_path: Path, filename: str) -> str:
    """Append _1, _2 suffixes on filename collision."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while (target_path / new_filename).exists():
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename


def random_copy_images(source: str, target: str, n: int = 100):
    """Recursively collect images from source and randomly copy n of them to target.

    Args:
        source: Source directory.
        target: Target directory.
        n: Number of images to pick.
    """
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()

    if not source_path.exists():
        print(f"❌ Source not found: {source_path}")
        return

    target_path.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Scanning {source_path} ...")

    # Recursively collect all images
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
    parser = argparse.ArgumentParser(description="Randomly pick and copy images")
    parser.add_argument("source", type=str, help="Source directory")
    parser.add_argument("target", type=str, help="Target directory")
    parser.add_argument("count", type=int, nargs="?", default=100, help="Number of images to pick")
    return parser.parse_args()


def main():
    args = parse_opt()
    random_copy_images(args.source, args.target, args.count)


if __name__ == "__main__":
    main()
