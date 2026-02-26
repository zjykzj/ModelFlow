# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/24 21:30
@File    : generate_calib_cache_for_coco.py
@Author  : zj
@Description: 
生成 COCO 目标检测模型 (如 YOLOv8) TensorRT INT8 校准所需的二进制缓存文件。

# 运行示例 (YOLOv8 标准 640x640):
# python3 generate_calib_cache_for_coco.py --input_dir export/coco_calib_src --output_dir export/coco_calib_dst --input_size 640

# 预期输出日志:
# 📂 找到 100 张图片，将处理前 100 张。
# ⚙️  预处理配置：LetterBox -> Resize(640x640) -> BGR2RGB -> Normalize
# --------------------------------------------------
#    已处理 100 / 100 ...
# --------------------------------------------------
# ✅ 完成！共成功保存 100 个文件到：export/coco_calib_dst

# 验证文件大小 (Float32):
# 公式：640 * 640 * 3 * 4 Bytes = 4,915,200 Bytes (~4.7 MB)
# $ ls -lh export/coco_calib_dst/*.bin | head -n 3
# -rw-r--r-- 1 root root 4.7M Feb 24 21:40 export/coco_calib_dst/000000000123.bin
# -rw-r--r-- 1 root root 4.7M Feb 24 21:40 export/coco_calib_dst/000000000456.bin
# -rw-r--r-- 1 root root 4.7M Feb 24 21:40 export/coco_calib_dst/000000000789.bin

# 💡 下一步命令示例:
# trtexec --onnx=yolov8.onnx --saveEngine=yolov8_int8.engine --int8 --calib=export/coco_calib_dst
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

# 假设该模块包含 YOLOv8 特定的预处理逻辑 (LetterBox, Normalize 等)
from core.npy.yolov8_preprocess import ImgPrepare


def main():
    parser = argparse.ArgumentParser(description="预处理 COCO 图片为 TensorRT INT8 校准所需的二进制格式")
    parser.add_argument("--input_dir", type=str, required=True, help="源图片文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出二进制文件的文件夹路径")
    # 检测模型通常使用 640 或 1280，默认为 640
    parser.add_argument("--input_size", type=int, default=640, help="目标输入尺寸 (默认 640，YOLOv8 标准)")
    parser.add_argument("--format", type=str, choices=["bin", "npy"], default="bin", help="输出文件格式 (bin 或 npy)")
    parser.add_argument("--max_images", type=int, default=100,
                        help="最大处理图片数量 (校准通常不需要太多，50-100 即可)")

    args = parser.parse_args()

    # 1. 定义 YOLOv8 预处理流程
    # 注意：ImgPrepare 具体实现应包含 LetterBox (保持宽高比填充) 逻辑
    transform = ImgPrepare(args.input_size, half=False)

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
    print(f"⚙️  预处理配置：LetterBox -> Resize({args.input_size}x{args.input_size}) -> BGR2RGB -> Normalize")
    print("-" * 50)

    count = 0
    for i, img_path in enumerate(image_files):
        try:
            # 打开图片
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("无法读取图片 (文件可能损坏或格式不支持)")

            # 执行预处理
            # 假设 transform 返回: (tensor, pad_info, scale_info, ...)
            # 我们只需要第一个返回值 (tensor) 用于校准
            result = transform(img)

            # 安全解包：如果返回的是元组则取第一个，否则直接使用
            if isinstance(result, tuple):
                data = result[0]
            else:
                data = result

            # 确保数据是连续的内存块 (tofile 需要)
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)

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
            if count % 10 == 0:  # 检测模型图片较大，降低日志频率
                print(f"   已处理 {count} / {len(image_files)} ...")

        except Exception as e:
            print(f"⚠️  跳过 {img_path.name}: {e}")

    print("-" * 50)
    print(f"✅ 完成！共成功保存 {count} 个文件到：{output_path}")

    if count > 0:
        # 计算理论文件大小供用户验证
        expected_size = args.input_size * args.input_size * 3 * 4
        print(f"💡 验证提示：每个 .bin 文件大小应为 {expected_size:,} Bytes ({expected_size / 1024 / 1024:.2f} MB)")
        print(f"💡 下一步命令示例:")
        print(f"   trtexec --onnx=model.onnx --saveEngine=model_int8.engine --int8 --calib={output_path}")


if __name__ == "__main__":
    main()