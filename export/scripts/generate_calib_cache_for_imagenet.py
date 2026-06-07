# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : generate_calib_cache_for_imagenet.py
@Author  : zj
@Description: 生成 ImageNet 分类模型 TensorRT INT8 校准数据

使用 export.core.utils 提供的预处理管线（零外部依赖）。

用法：
    python3 export/scripts/generate_calib_cache_for_imagenet.py \\
        --input_dir /path/to/imagenet_val \\
        --output_dir ./calib_imagenet_dst

输出：
    .bin 文件，每个文件包含预处理后的 float32 张量
    文件大小：224 * 224 * 3 * 4 = 602,112 Bytes
"""

import argparse
from pathlib import Path

import cv2

from export.core.utils import classify_preprocess, save_calib_data, collect_images


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet INT8 校准数据生成"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="源图片目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--crop_size", type=int, default=224, help="裁剪尺寸")
    parser.add_argument("--resize_size", type=int, default=256, help="缩放尺寸")
    parser.add_argument("--format", type=str, choices=["bin", "npy"], default="bin")
    parser.add_argument("--max_images", type=int, default=100,
                        help="最大处理图片数（校准通常 50-100 即可）")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集图片
    image_files = collect_images(input_path, max_images=args.max_images)
    if not image_files:
        print(f"❌ No images found in {input_path}")
        return

    print(f"📂 Found {len(image_files)} images")
    print(f"⚙️  Preprocessing: Resize({args.resize_size}) -> CenterCrop({args.crop_size}) -> Normalize")
    print("-" * 50)

    count = 0
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Cannot read image")

            # 使用 export 自包含预处理
            data = classify_preprocess(img, args.crop_size, args.resize_size)

            save_calib_data(data, output_path, img_path.stem, args.format)
            count += 1

            if count % 100 == 0:
                print(f"  Processed {count} / {len(image_files)} ...")

        except Exception as e:
            print(f"  ⚠️  Skip {img_path.name}: {e}")

    print("-" * 50)
    print(f"✅ Done! {count} files saved to {output_path}")

    if count > 0:
        expected = args.crop_size * args.crop_size * 3 * 4
        print(f"💡 Expected file size: {expected:,} Bytes ({expected / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
