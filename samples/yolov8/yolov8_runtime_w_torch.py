# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/30 15:57
@File    : yolov8_runtime_w_torch.py
@Author  : zj
@Description: 
"""

import cv2
import time
import sys
import logging

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

from tqdm import tqdm
from pathlib import Path
from typing import Union, Tuple, Optional

# ----------------------------
# 配置日志（必须放在最前面）
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

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

# ----------------------------
# 导入模块
# ----------------------------

from core.backends.backend_runtime import BackendRuntime
from general import CLASSES_NAME, draw_results
from numpy_util import letterbox
from torch_util import non_max_suppression, scale_boxes


# ----------------------------
# 预处理与后处理函数
# ----------------------------

def preprocess(im0: ndarray, img_size: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False) -> ndarray:
    """
    图像预处理：缩放、转格式、归一化、添加 batch 维度。
    """
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255.0  # 0-255 to 0.0-1.0
    if im.ndim == 3:
        im = im[None]  # expand for batch dim
    return im


def postprocess(
        preds: ndarray,
        im_shape: tuple,  # (h, w) of input to model
        im0_shape: tuple,  # (h, w) of original image
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[list] = None,
        agnostic: bool = False,
        max_det: int = 300,
) -> tuple:
    """
    后处理：NMS + 坐标缩放。
    Returns:
        boxes, confs, cls_ids
    """
    pred = non_max_suppression(preds, conf, iou, classes, agnostic, max_det=max_det)[0]
    boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
    confs = pred[:, 4:5]
    cls_ids = pred[:, 5:6]
    return boxes, confs, cls_ids


class YOLOv8Runtime:
    def __init__(self, weight: str = 'yolov8s.onnx'):
        super().__init__()
        self.session = BackendRuntime(weight)
        self.session.load()

        input_name = self.session.get_input_names()[0]
        self.net_h, self.net_w = self.session.get_input_shapes()[input_name][2:]

        if self.session.providers[0] == 'CUDAExecutionProvider':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def infer(self, im: ndarray) -> Tensor:
        input_name = self.session.input_names[0]
        output_dict = self.session.infer({input_name: im})
        output_name = self.session.output_names[0]
        return self.from_numpy(output_dict[output_name])

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.45) -> tuple:
        """
        检测图像中的目标，并统计各阶段耗时。
        Returns:
            boxes, confs, cls_ids
        """
        # 记录开始时间
        t0 = time.perf_counter()

        # --- 预处理 ---
        t_pre_start = time.perf_counter()
        im = preprocess(im0, (self.net_h, self.net_w))
        t_pre_end = time.perf_counter()

        # --- 推理 ---
        t_inf_start = time.perf_counter()
        outputs = self.infer(im)
        t_inf_end = time.perf_counter()

        # --- 后处理 ---
        t_post_start = time.perf_counter()
        boxes, confs, cls_ids = postprocess(
            outputs,
            im.shape[2:],  # 模型输入尺寸 (h, w)
            im0.shape[:2],  # 原图尺寸 (h, w)
            conf=conf,
            iou=iou
        )
        t_post_end = time.perf_counter()

        # --- 耗时统计 ---
        pre_time = (t_pre_end - t_pre_start) * 1000  # ms
        inf_time = (t_inf_end - t_inf_start) * 1000
        post_time = (t_post_end - t_post_start) * 1000
        total_time = (t_post_end - t0) * 1000

        logging.info(
            f"Detect time - Pre: {pre_time:.2f}ms | "
            f"Infer: {inf_time:.2f}ms | "
            f"Post: {post_time:.2f}ms | "
            f"Total: {total_time:.2f}ms"
        )

        return boxes, confs, cls_ids


# ----------------------------
# 独立的推理函数（增强：支持耗时统计）
# ----------------------------

def predict_image(
        model: YOLOv8Runtime,
        img_path: Union[str, Path],
        output_dir: Union[str, Path] = "output",
        suffix: str = "yolov8",
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
        model: YOLOv8Runtime,
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
        description="YOLOv8 ONNX Runtime 推理工具",
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
        default="yolov8",
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
    model = YOLOv8Runtime(args.model)

    if args.mode == 'image':
        predict_image(model, args.input, args.output_dir, args.suffix, args.save)
    else:
        predict_video(model, args.input, args.output_dir, args.suffix, args.save)


if __name__ == '__main__':
    main()
