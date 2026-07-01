# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : generate_calib_cache_for_imagenet.py
@Author  : zj
@Description: Generate ImageNet classification model TensorRT INT8 calibration data

Uses the preprocessing pipeline provided by export._utils (zero external dependencies).

Usage:
    python3 export/tensorrt/scripts/generate_calib_cache_for_imagenet.py \\
        --input_dir /path/to/imagenet_val \\
        --output_dir ./calib_imagenet_dst

Output:
    .bin files, each containing a preprocessed float32 tensor.
    File size: 224 * 224 * 3 * 4 = 602,112 Bytes.
"""

import argparse
from pathlib import Path

import cv2

from export._utils import classify_preprocess, save_calib_data, collect_images


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet INT8 calibration data generator"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Source image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--crop_size", type=int, default=224, help="Crop size")
    parser.add_argument("--resize_size", type=int, default=256, help="Resize size")
    parser.add_argument("--format", type=str, choices=["bin", "npy"], default="bin")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Maximum number of images to process (50-100 is usually sufficient for calibration)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect images
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

            # Use export's self-contained preprocessing
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
