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
from typing import Union, Tuple, Optional, Any, List

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
from torch_util import non_max_suppression, scale_boxes, process_mask


# ----------------------------
# 预处理与后处理函数
# ----------------------------

def preprocess(im0: ndarray, img_size: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False) -> ndarray:
    """
    图像预处理：缩放、转格式、归一化、添加 batch 维度。
    """
    # im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im, ratio, (dw, dh) = letterbox(im0, img_size, stride=stride, auto=auto)  # padded resize
    print(f"ratio: {ratio} - dw: {dw} - dh: {dh}")
    cv2.imwrite("imimmm.jpg", im)
    imm = np.copy(im)
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = im.astype(np.float32)
    im /= 255.0  # 0-255 to 0.0-1.0
    if im.ndim == 3:
        im = im[None]  # expand for batch dim
    return im, imm, ratio, (dw, dh)


import torch
import torch.nn.functional as F


def resize_and_scale_masks(masks, img1_shape, img0_shape, ratio, padding):
    """
    将模型输出的 masks (小图) 缩放到原始图像 (img0) 的尺寸。

    Args:
        masks (torch.Tensor): [N, H_small, W_small], 模型输出的二值 mask (0/1)
        img1_shape (tuple): 模型输入图像的形状 (H, W)，即 padded 后的图像
        img0_shape (tuple): 原始图像的形状 (H, W)
        ratio (tuple): 缩放比例 (ratio_h, ratio_w)
        padding (tuple): 填充值 (dw, dh)

    Returns:
        torch.Tensor: [N, img0_shape[0], img0_shape[1]], 缩放后的二值 mask
    """
    r = ratio[0]
    new_unpad = int(round(img0_shape[1] * r)), int(round(img0_shape[0] * r))

    dw, dh = padding
    print(f"ratio: {ratio} - dw: {dw} - dh: {dh}")
    print(f"new_unpad: {new_unpad}")

    assert (new_unpad[0] + 2 * dw) == img1_shape[1]
    assert (new_unpad[1] + 2 * dh) == img1_shape[0]

    dw = int(dw)
    dh = int(dh)
    masks_unpad = masks[:, dh:(dh + new_unpad[1]), dw:(dw + new_unpad[0])]
    print(f"masks_unpad.shape: {masks_unpad.shape}")

    # Step 1: 将 mask 插值到填充后的尺寸
    scaled_masks = F.interpolate(
        masks_unpad.unsqueeze(1),  # [N, 1, H_small, W_small]
        size=img0_shape,  # 目标是填充后的尺寸
        mode='nearest',
        # align_corners=False
    ).squeeze(1).float()  # [N, H_padded, W_padded]

    # # Step 2: 创建一个与原始图像等大的 mask，并将插值后的 mask 放入有效区域
    # final_masks = torch.zeros((masks.shape[0], img0_shape[0], img0_shape[1]),
    #                           dtype=masks.dtype, device=masks.device)
    #
    # # 计算有效区域（去除 padding 后的区域）
    # valid_h = int(img0_shape[0] * ratio[0])
    # valid_w = int(img0_shape[1] * ratio[1])
    #
    # # 考虑填充量
    # x_start = int(padding[0])
    # y_start = int(padding[1])
    #
    # # 确保不会超出边界
    # x_end = min(x_start + valid_w, img1_shape[1])
    # y_end = min(y_start + valid_h, img1_shape[0])
    #
    # valid_w = x_end - x_start
    # valid_h = y_end - y_start
    #
    # # 将插值后的 mask 放入大 mask 的对应位置
    # final_masks[:, :valid_h, :valid_w] = scaled_masks[:, y_start:y_end, x_start:x_end]

    # Step 3: 二值化（可选，如果需要严格 0/1）
    # final_masks = (final_masks > 0.5).float()
    final_masks = (scaled_masks > 0.5).float()

    return final_masks


import numpy as np
import cv2


def save_empty_masks(masks, imm, output_dir='./output'):
    """
    在原始图像上应用掩码，并保存结果图像。

    参数:
    - masks: 掩码数组，形状为 [N, H, W]，其中 N 是掩码数量，H 和 W 分别是高度和宽度。
    - imm: 原始图像，形状为 [H, W, 3]。
    - output_dir: 结果图像保存目录。
    """
    height, width = masks.shape[-2:]
    assert imm.shape[:2] == (height, width), "Mask and image dimensions do not match."

    # 确保输出目录存在
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(masks.shape[0]):
        mask = masks[i].numpy()  # 将当前掩码转换为numpy数组
        mask = np.uint8(mask * 255)  # 将其从{0,1}转化为{0,255}范围内的值

        # 创建彩色掩码：将单通道掩码扩展到三通道
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 应用掩码到原始图像
        result_image = imm.copy()
        # 将掩码区域设为红色（或其他颜色），这里使用了一个简单的例子
        red_color = np.array([0, 0, 255], dtype=np.uint8)
        result_image[(mask > 0)] = red_color

        # 保存结果图像
        output_path = os.path.join(output_dir, f'result_mask_{i + 1}.png')
        cv2.imwrite(output_path, result_image)


def masks2segments(masks, strategy: str = "all"):
    """
    Convert masks to segments using contour detection.

    Args:
        masks (torch.Tensor): Binary masks with shape (batch_size, 160, 160).
        strategy (str): Segmentation strategy, either 'all' or 'largest'.

    Returns:
        (list): List of segment masks as float32 arrays.
    """
    from ultralytics.data.converter import merge_multi_segment

    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # merge and concatenate all segments
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def clip_coords(coords, shape):
    """
    Clip line coordinates to image boundaries.

    Args:
        coords (torch.Tensor | np.ndarray): Line coordinates to clip.
        shape (tuple): Image shape as (height, width).

    Returns:
        (torch.Tensor | np.ndarray): Clipped coordinates.
    """
    if isinstance(coords, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize: bool = False, padding: bool = True):
    """
    Rescale segment coordinates from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): Shape of the source image.
        coords (torch.Tensor): Coordinates to scale with shape (N, 2).
        img0_shape (tuple): Shape of the target image.
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_h, pad_w)).
        normalize (bool): Whether to normalize coordinates to range [0, 1].
        padding (bool): Whether coordinates are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


import cv2
import numpy as np


def draw_segmentation_masks(image, mask_res, color=(0, 255, 0), alpha=0.5):
    """
    在原图上绘制分割掩码。

    参数:
    - image: 原始图像 (BGR 格式)。
    - mask_res: 分割结果列表，每个元素是一个形状为 (N, 2) 的数组，
                其中 N 是多边形顶点数，每一行是一对 (x, y) 坐标。
    - color: 掩码颜色，默认为绿色。
    - alpha: 掩码透明度，范围 [0, 1]。

    返回:
    - 绘制了分割掩码后的图像。
    """
    overlay = image.copy()
    output = image.copy()

    for mask in mask_res:
        # 确保有至少3个点才能构成一个多边形
        if mask.shape[0] >= 3:
            pts = mask.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color=color)

    # 应用透明度
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def postprocess(
        imm,
        im0, ratio, padding,
        preds: List[Any],
        im_shape: tuple,  # (h, w) of input to model
        im0_shape: tuple,  # (h, w) of original image
        conf: float = 0.25,
        iou: float = 0.7,
        classes: Optional[list] = None,
        agnostic: bool = False,
        max_det: int = 300,
) -> tuple:
    """
    后处理：NMS + 坐标缩放。
    Returns:
        boxes, confs, cls_ids
    """
    protos = preds[1]
    print(f"protos shape: {protos.shape}")
    preds = preds[0]
    print(f"preds shape: {preds.shape}")

    preds = non_max_suppression(
        preds,
        0.5,
        0.7,
        classes,
        agnostic,
        max_det=max_det,
        nc=80,
        end2end=False,
        rotated=False,
        return_idxs=False,
    )
    print(f"preds len: {len(preds)} - preds[0].shape = {preds[0].shape}")
    preds = preds[0]

    masks = process_mask(protos[0], preds[:, 6:], preds[:, :4], im_shape, upsample=True)  # HWC
    preds[:, :4] = scale_boxes(im_shape, preds[:, :4], im0_shape)
    print(f"masks shape: {masks.shape}")
    print(f"preds shape: {preds.shape}")

    print(f"im_shape: {im_shape} - im0_shape: {im0_shape}")
    # masks = resize_binary_masks(masks, im0_shape)
    # save_empty_masks(masks, imm)
    # masks = resize_and_scale_masks(masks, im_shape, im0_shape, ratio, padding)
    # save_empty_masks(masks, im0)

    mask_res = [
        scale_coords(masks.shape[1:], x, im0_shape, normalize=False)
        for x in masks2segments(masks)
    ]
    print(f"len(mask_res): {len(mask_res)} - mask_res[0].shape = {mask_res[0].shape}")

    im0_mask = draw_segmentation_masks(im0, mask_res)
    cv2.imwrite("output/im0_mask.jpg", im0_mask)

    # if masks is not None:
    #     keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
    #     preds, masks = preds[keep], masks[keep]
    #     print(f"masks shape: {masks.shape}")
    #     print(f"preds shape: {preds.shape}")
    # print(f"masks type: {type(masks)} - masks[0][0][0] type: {type(masks[0][0][0])}")
    # print(f"max masks: {masks.max()}")

    boxes = preds[:, :4]
    confs = preds[:, 4:5]
    cls_ids = preds[:, 5:6]
    return boxes, confs, cls_ids, masks


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

    def infer(self, im: ndarray) -> list[Any]:
        input_name = self.session.input_names[0]
        output_dict = self.session.infer({input_name: im})

        preds = []
        for output_name in self.session.output_names:
            pred = output_dict[output_name]
            print(output_name, pred.shape)
            preds.append(self.from_numpy(pred))

        return preds

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.7) -> tuple:
        """
        检测图像中的目标，并统计各阶段耗时。
        Returns:
            boxes, confs, cls_ids
        """
        # 记录开始时间
        t0 = time.perf_counter()

        # --- 预处理 ---
        t_pre_start = time.perf_counter()
        im, imm, ratio, padding = preprocess(im0, (self.net_h, self.net_w))
        t_pre_end = time.perf_counter()

        # --- 推理 ---
        t_inf_start = time.perf_counter()
        preds = self.infer(im)
        t_inf_end = time.perf_counter()

        # --- 后处理 ---
        t_post_start = time.perf_counter()
        boxes, confs, cls_ids, masks = postprocess(
            imm,
            im0, ratio, padding,
            preds,
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

        return boxes, confs, cls_ids, masks


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

    boxes, confs, cls_ids, masks = model.detect(im0)
    # overlay = draw_results(im0, boxes, confs, cls_ids, CLASSES_NAME, is_xyxy=True)

    # 调用绘图函数（自动处理缩放）
    overlay = draw_results(
        img=im0,
        boxes=boxes,
        confs=confs,
        cls_ids=cls_ids,
        masks=masks,  # 自动判断是否需要缩放
        CLASSES_NAME=CLASSES_NAME,
        alpha=0.5,
        is_xyxy=True
    )

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
