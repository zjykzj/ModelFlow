# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/24 20:27
@File    : random_copy_images.py
@Author  : zj
@Description:

作用：从源目录递归扫描所有图片，随机抽取指定数量（N_SAMPLES）并复制到目标目录。若文件名冲突，自动添加后缀（如 `_1`）防止覆盖。

示例：
- 场景：从 COCO 验证集 (`val2017`) 中随机挑 100 张图用于快速测试。
- 配置：设 `SOURCE_DIR` 为数据集路径，`N_SAMPLES = 100`。
- 结果：当前目录下生成 `cal_src_for_coco_det` 文件夹，内含 100 张随机图片。

"""

import os
import shutil
import random
from pathlib import Path

# ================= 配置区域 =================
# 源目录：包含所有子目录的根路径
# SOURCE_DIR = r'/path/to/source/folder'
# SOURCE_DIR = r'/home/zjykzj/datasets/imagenet/val'
SOURCE_DIR = r'/workdir/datasets/coco/images/val2017'

# 目标目录：存放随机抽取图片的路径
# TARGET_DIR = r'./cal_src_for_imagenet'
TARGET_DIR = r'./cal_src_for_coco_det'

# 需要随机抽取的图片数量
N_SAMPLES = 100

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.ico', '.svg'}


# ===========================================

def get_unique_filename(target_path, filename):
    """如果文件名冲突，添加后缀 _1, _2 等"""
    base_name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(target_path, new_filename)):
        new_filename = f"{base_name}_{counter}{ext}"
        counter += 1

    return new_filename


def random_copy_images(source, target, n):
    source_path = Path(source).resolve()
    target_path = Path(target).resolve()

    if not source_path.exists():
        print(f"❌ 错误：源目录不存在 -> {source_path}")
        return

    # 创建目标目录
    try:
        target_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"❌ 错误：无法创建目标目录 -> {target_path}")
        return

    print(f"🔍 正在扫描源目录：{source_path} ...")

    # 1. 深度遍历收集所有图片路径
    all_image_paths = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                all_image_paths.append(os.path.join(root, file))

    total_found = len(all_image_paths)
    print(f"✅ 共发现 {total_found} 张图片。")

    if total_found == 0:
        print("⚠️ 未找到任何图片，操作结束。")
        return

    # 2. 确定实际要复制的数量
    if n > total_found:
        print(f"⚠️ 警告：请求复制 {n} 张，但只有 {total_found} 张。将复制所有图片。")
        n = total_found

    # 3. 随机采样 (不重复)
    selected_images = random.sample(all_image_paths, n)
    print(f"🎲 已随机选中 {len(selected_images)} 张图片，开始复制...")

    count_success = 0
    count_fail = 0

    for src_file in selected_images:
        filename = os.path.basename(src_file)
        # 处理重名
        final_name = get_unique_filename(target_path, filename)
        dst_file = os.path.join(target_path, final_name)

        try:
            shutil.copy2(src_file, dst_file)
            # 可选：打印详细日志，如果图片太多可以注释掉下面这行
            # print(f"  [OK] {filename} -> {final_name}")
            count_success += 1
        except Exception as e:
            print(f"  [FAIL] 复制失败 {filename}: {e}")
            count_fail += 1

    print("-" * 30)
    print(f"🎉 完成！")
    print(f"   计划复制：{n} 张")
    print(f"   成功复制：{count_success} 张")
    if count_fail > 0:
        print(f"   失败数量：{count_fail} 张")
    print(f"   结果保存至：{target_path}")


if __name__ == "__main__":
    # 如果你希望直接运行脚本而不修改代码，可以使用命令行参数
    # 这里为了简单，直接读取顶部的配置变量
    random_copy_images(SOURCE_DIR, TARGET_DIR, N_SAMPLES)

    # --- 进阶：支持命令行参数运行 (取消下方注释即可使用) ---
    # import argparse
    # parser = argparse.ArgumentParser(description="从目录随机复制 N 张图片")
    # parser.add_argument("source", help="源目录")
    # parser.add_argument("target", help="目标目录")
    # parser.add_argument("count", type=int, help="随机抽取的数量")
    # args = parser.parse_args()
    # random_copy_images(args.source, args.target, args.count)
