# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/3 20:38
@File    : main.py
@Author  : zj
@Description: 
"""

import cv2
import sys
import time
import logging

from tqdm import tqdm
from typing import Union
from pathlib import Path

# ----------------------------
# 项目路径设置
# ----------------------------

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

CURRENT_DIR = Path.cwd()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from general import CLASSES_NAME, draw_results
from yolov5_runtime_w_numpy import YOLOv5Runtime


# ----------------------------
# 独立的推理函数（增强：支持耗时统计）
# ----------------------------

def predict_image(
        model: YOLOv5Runtime,
        img_path: Union[str, Path],
        output_dir: Union[str, Path] = "output",
        suffix: str = "yolov5",
        save: bool = False
):
    """
    预测单张图像，并打印端到端耗时。
    """
    img_path = Path(img_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    im0 = cv2.imread(str(img_path))
    if im0 is None:
        raise ValueError(f"无法读取图像: {img_path}")

    logging.info(f"正在处理图像: {img_path.name}")

    # 端到端耗时（包含读图 + 推理 + 绘图）
    t_start = time.time()

    boxes, confs, cls_ids = model.detect(im0)
    overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)

    t_end = time.time()
    total_time = (t_end - t_start) * 1000  # ms
    fps = 1000 / total_time if total_time > 0 else 0

    logging.info(f"检测到 {len(boxes)} 个目标 | 耗时: {total_time:.2f}ms | FPS: {fps:.1f}")

    if save:
        save_path = output_dir / f"{img_path.stem}-{suffix}.jpg"
        cv2.imwrite(str(save_path), overlay)
        logging.info(f"结果已保存至: {save_path}")


def predict_video(
        model: YOLOv5Runtime,
        video_file: Union[str, Path],
        output_dir: Union[str, Path] = "output",
        suffix: str = "yolov5",
        save: bool = False
):
    """
    预测视频，支持保存，打印平均 FPS。
    """
    video_path = Path(video_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"视频信息: {video_path.name} | {fps:.1f} FPS | {total_frames} 帧 | {frame_width}x{frame_height}")

    writer = None
    if save:
        save_path = output_dir / f"{video_path.stem}-{suffix}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (frame_width, frame_height))

    # 统计总耗时
    total_infer_time = 0.0
    processed_frames = 0

    try:
        for _ in tqdm(range(total_frames), desc="处理视频"):
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()
            boxes, confs, cls_ids = model.detect(frame)
            overlay = draw_results(frame, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)
            t_end = time.time()

            frame_time = (t_end - t_start) * 1000
            total_infer_time += frame_time
            processed_frames += 1

            if save and writer:
                writer.write(overlay)

    finally:
        cap.release()
        if writer:
            writer.release()

    # 打印平均性能
    avg_time = total_infer_time / processed_frames if processed_frames > 0 else 0
    avg_fps = 1000 / avg_time if avg_time > 0 else 0
    logging.info(f"平均耗时: {avg_time:.2f}ms/帧 | 平均 FPS: {avg_fps:.1f}")

    if save:
        logging.info(f"视频已保存至: {save_path}")


# ----------------------------
# 命令行参数解析（智能识别输入类型）
# ----------------------------

def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLOv5 Infer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model",
        type=str,
        help="ONNX 模型路径 (e.g., yolov5s.onnx)"
    )
    parser.add_argument(
        "input",
        type=str,
        help="输入图像或视频路径"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存结果到输出目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="yolov5",
        help="输出文件名后缀"
    )

    args = parser.parse_args()

    # 自动判断是图像还是视频（避免手动加 --video）
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"输入文件不存在: {args.input}")

    # 常见图像/视频扩展名
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

    if input_path.suffix.lower() in image_exts:
        args.mode = 'image'
    elif input_path.suffix.lower() in video_exts:
        args.mode = 'video'
    else:
        parser.error(f"不支持的文件格式: {input_path.suffix}")

    logging.info(f"解析参数: {args}")
    return args


# ----------------------------
# 主函数
# ----------------------------

def main():
    args = parse_opt()

    from yolov5_runtime_w_numpy import YOLOv5Runtime
    model = YOLOv5Runtime(args.model)

    if args.mode == 'image':
        predict_image(model, args.input, args.output_dir, args.suffix, args.save)
    else:
        predict_video(model, args.input, args.output_dir, args.suffix, args.save)


if __name__ == '__main__':
    main()
