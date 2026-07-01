# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : generate_calib_cache_for_coco.py
@Author  : zj
@Description: Generate COCO detection model TensorRT INT8 calibration data

Uses the preprocessing pipeline provided by export._utils (zero external dependencies).

Usage:
    python3 export/tensorrt/scripts/generate_calib_cache_for_coco.py \\
        --input_dir /path/to/coco_images \\
        --output_dir ./calib_coco_dst

Output:
    .bin files, each containing a preprocessed float32 tensor
    File size: 640 * 640 * 3 * 4 = 4,915,200 Bytes (~4.7 MB)
"""

import argparse
from pathlib import Path

import cv2

from export._utils import detect_preprocess, save_calib_data, collect_images


def main():
    parser = argparse.ArgumentParser(
        description="COCO INT8 calibration data generator"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Source image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--input_size", type=int, default=640, help="Target input size")
    parser.add_argument("--format", type=str, choices=["bin", "npy"], default="bin")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Max images to process (50-100 is usually sufficient for calibration)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect images
    image_files = collect_images(input_path, max_images=args.max_images)
    if not image_files:
        print(f"No images found in {input_path}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Preprocessing: LetterBox({args.input_size}) -> BGR2RGB -> /255")
    print("-" * 50)

    count = 0
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Cannot read image")

            # Use export self-contained preprocessing
            data, scale, pad = detect_preprocess(img, args.input_size)

            save_calib_data(data, output_path, img_path.stem, args.format)
            count += 1

            if count % 10 == 0:
                print(f"  Processed {count} / {len(image_files)} ...")

        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! {count} files saved to {output_path}")

    if count > 0:
        expected = args.input_size * args.input_size * 3 * 4
        print(f"Expected file size: {expected:,} Bytes ({expected / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
