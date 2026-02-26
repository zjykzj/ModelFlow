# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/24 20:43
@File    : generate_calib_cache_for_classify.py
@Author  : zj
@Description:
生成 TensorRT INT8 校准所需的二进制缓存文件。

# 运行示例：
# python3 generate_calib_cache_for_classify.py --input_dir export/cal_imagenet_src --output_dir export/cal_imagenet_dst --crop_size 224

# 预期输出日志：
# 📂 找到 100 张图片，将处理前 100 张。
# ⚙️  预处理配置：Resize(256) -> CenterCrop(224x224) -> Normalize
# --------------------------------------------------
#    已处理 100 / 100 ...
# --------------------------------------------------
# ✅ 完成！共成功保存 100 个文件到：export/cal_imagenet_dst

# 验证文件大小 (Float32):
# 公式：224 * 224 * 3 * 4 Bytes = 602,112 Bytes
# $ ls -l export/cal_imagenet_dst/*.bin | head -n 3
# -rw-r--r-- 1 root root 602112 Feb 24 20:56 export/cal_imagenet_dst/ILSVRC2012_val_00000243.bin
# -rw-r--r-- 1 root root 602112 Feb 24 20:56 export/cal_imagenet_dst/ILSVRC2012_val_00000616.bin
# -rw-r--r-- 1 root root 602112 Feb 24 20:56 export/cal_imagenet_dst/ILSVRC2012_val_00000676.bin
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

# 假设该模块在当前环境下可用
from core.npy.classify_preprocess import ImgPrepare


def main():
    parser = argparse.ArgumentParser(description="预处理图片为 TensorRT INT8 校准所需的二进制格式")
    parser.add_argument("--input_dir", type=str, required=True, help="源图片文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出二进制文件的文件夹路径")
    parser.add_argument("--crop_size", type=int, default=224, help="目标裁剪尺寸 (默认 224)")
    parser.add_argument("--format", type=str, choices=["bin", "npy"], default="bin", help="输出文件格式 (bin 或 npy)")
    parser.add_argument("--max_images", type=int, default=100,
                        help="最大处理图片数量 (校准通常不需要太多，50-100 即可)")

    args = parser.parse_args()

    # 1. 定义 ImageNet 预处理流程
    # 注意：这里硬编码了 input_size=256，意味着流程是 Resize(256) -> Crop(args.crop_size)
    transform = ImgPrepare(input_size=256, crop_size=args.crop_size, batch=False, mode="crop")

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取图片列表
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in Path(args.input_dir).iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"❌ 在 {args.input_dir} 中未找到任何图片。")
        return

    # 限制数量
    image_files = image_files[:args.max_images]
    print(f"📂 找到 {len(image_files)} 张图片，将处理前 {len(image_files)} 张。")

    # 【修正】补全了配置打印语句，使其与 Docstring 中的日志示例一致
    # 根据代码 input_size=256 和 crop_size 推断实际流程
    print(f"⚙️  预处理配置：Resize(256) -> CenterCrop({args.crop_size}x{args.crop_size}) -> Normalize")
    print("-" * 50)

    count = 0
    for i, img_path in enumerate(image_files):
        try:
            # 打开图片
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("无法读取图片 (可能是损坏的文件)")

            # 执行预处理
            data = transform(img)

            # 确定输出文件名和路径
            stem = img_path.stem
            if args.format == 'bin':
                out_file = output_path / f"{stem}.bin"
                # tofile() 保存为原始二进制字节 (Raw Binary)
                data.tofile(out_file)
            else:  # npy
                out_file = output_path / f"{stem}.npy"
                np.save(out_file, data)

            count += 1
            if count % 100 == 0:
                print(f"   已处理 {count} / {len(image_files)} ...")

        except Exception as e:
            print(f"⚠️  跳过 {img_path.name}: {e}")

    print("-" * 50)
    print(f"✅ 完成！共成功保存 {count} 个文件到：{output_path}")
    if count > 0:
        print(f"💡 下一步命令示例:")
        print(f"   trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8 --calib={output_path}")


if __name__ == "__main__":
    main()