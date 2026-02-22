# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 15:00
@File    : yolov8seg_evaluator.py
@Author  : zj
@Description: YOLOv8-seg 实例分割评估器（COCO mAP 格式）
"""

import os
import json
import time
import cv2

import numpy as np
from numpy import ndarray
from typing import Union, Tuple, Optional, Any, List

from tqdm import tqdm
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from core.npy.v8.postprocessor import non_max_suppression
from core.npy.v8.ops import scale_boxes, process_mask, scale_image

from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)


def postprocess(
        pred: List[ndarray],
        im_shape: Tuple,  # (h, w) of input to model
        im0_shape: Tuple,  # (h, w) of original image
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[list] = None,
        agnostic: bool = False,
        max_det: int = 300,
        nc: int = 0,  # number of classes (optional)
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Postprocessing: NMS + coordinate scaling.
    Returns:
        boxes (np.ndarray): Scaled bounding boxes in xyxy format, shape (N, 4)
        confs (np.ndarray): Confidence scores, shape (N, 1)
        cls_ids (np.ndarray): Class IDs, shape (N, 1)
        masks (np.ndarray): Segmentation masks for each detection, shape (N, 1, 1)
     """
    proto = pred[1][-1] if isinstance(pred[1], tuple) else pred[1]
    proto = proto[0]  # [1, 32, 160, 160] -> [32, 160, 160]

    pred = non_max_suppression(
        pred[0],
        conf,
        iou,
        classes=classes,
        agnostic=agnostic,
        max_det=max_det,
        nc=nc,
    )
    pred = pred[0]  # [1, 300, 6] -> [300, 6]

    if len(pred) > 0:
        masks = process_mask(proto, pred[:, 6:], pred[:, :4], im_shape, upsample=True)

        masks = scale_image(np.transpose(masks, (1, 2, 0)), im0_shape)
        masks = np.transpose(masks, (2, 0, 1))

        boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
        confs = pred[:, 4:5]
        cls_ids = pred[:, 5:6]
    else:
        # ✅ 返回二维空数组，保持 shape 一致性
        boxes = np.zeros((0, 4), dtype=np.float32)
        confs = np.zeros((0, 1), dtype=np.float32)
        cls_ids = np.zeros((0, 1), dtype=np.float32)
        masks = np.zeros((0, 1, 1), dtype=np.float32)
    return boxes, confs, cls_ids, masks


###################################################### DRAW

def generate_colors(i, bgr=False):
    """生成可视化颜色"""
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color


def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255),
                       font=cv2.FONT_HERSHEY_COMPLEX, segment=None):
    """在图像上绘制检测框和标签"""
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

    if segment is not None:
        colr = np.asarray(color)
        colr = colr.reshape(1, 3).repeat((image.shape[0] * image.shape[1]), axis=0).reshape(image.shape[0],
                                                                                            image.shape[1], 3)
        image = cv2.addWeighted(image, 1, (colr * segment.reshape(*segment.shape[:2], 1)).astype(image.dtype), 0.8, 1)

    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
    return image


class EvalEvaluator:
    """
    YOLOv8-seg 实例分割评估器
    """

    def __init__(self, model, dataset, transform, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """
        Args:
            model: 分割模型，需包含 class_list 属性
            dataset: 评估数据集，返回 (img, path, img_h, img_w)
            transform: 预处理函数
            conf_thres: 置信度阈值
            iou_thres: NMS IOU 阈值
            max_det: 最大检测数量
        """
        self.model = model
        self.dataset = dataset
        self.transform = transform
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.class_list = model.class_list
        self.nc = len(self.class_list)
        self.stride = getattr(model, 'stride', 32)

    def run(self, save=False, save_dir='results'):
        """
        运行评估，生成 COCO 格式结果

        Args:
            save: 是否保存可视化结果
            save_dir: 可视化结果保存目录

        Returns:
            coco_results: list, COCO 格式检测结果（包含 bbox 和 segmentation）
        """
        total_preprocess_time = 0.0
        total_infer_time = 0.0
        total_postprocess_time = 0.0
        total_pipeline_time = 0.0
        coco_results = []

        if save:
            os.makedirs(save_dir, exist_ok=True)

        total = len(self.dataset)
        for idx in tqdm(range(total), desc="Evaluating YOLOv8-seg"):
            img_src, path, _ = self.dataset[idx]

            # --- 预处理 ---
            t1 = time.time()
            im, ratio, padding, im_shape, im0_shape = self.transform(img_src)
            t2 = time.time()

            # --- 推理 ---
            pred_results = self.model(im)
            t3 = time.time()

            # --- 后处理 ---
            # YOLOv8-seg 输出：[pred(1, 116, 8400), proto(1, 32, 160, 160)]
            pred = pred_results[0]
            proto = pred_results[1] if len(pred_results) > 1 else None

            boxes, scores, labels, masks = postprocess(
                pred_results,
                im_shape,
                im0_shape,
                conf=self.conf_thres,
                iou=self.iou_thres,
                nc=self.nc,
            )
            t4 = time.time()

            # --- 计时统计 ---
            total_preprocess_time += t2 - t1
            total_infer_time += t3 - t2
            total_postprocess_time += t4 - t3
            total_pipeline_time += t4 - t1

            # --- 生成 COCO 格式结果 ---
            if len(boxes) > 0:
                image_id = int(Path(path).stem) if Path(path).stem.isdigit() else idx + 1

                for box, score, label, mask in zip(boxes, scores, labels, masks):
                    # COCO bbox format: [x, y, width, height]
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    coco_box = [float(x1), float(y1), float(w), float(h)]

                    # COCO category_id 从 1 开始
                    category_id = int(label) + 1

                    # COCO segmentation format: RLE 或 polygon
                    # 这里使用 RLE 格式
                    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('utf-8')  # 转为字符串以便 JSON 序列化

                    coco_results.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": coco_box,
                        "segmentation": rle,
                        "score": float(score)
                    })

                # --- 可视化保存 ---
                if save:
                    img_ori = img_src.copy()
                    for box, score, label, mask in zip(boxes, scores, labels, masks):
                        class_num = int(label)
                        label_text = f'{self.class_list[class_num]} {float(score):.2f}'
                        color = generate_colors(class_num, bgr=True)

                        img_ori = plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), box,
                                                     label_text,
                                                     color=color, segment=mask)

                    save_path = os.path.join(save_dir, f"seg-{Path(path).stem}.jpg")
                    cv2.imwrite(save_path, img_ori)

        # --- 日志统计 ---
        avg_pre = total_preprocess_time / total if total else 0
        avg_inf = total_infer_time / total if total else 0
        avg_post = total_postprocess_time / total if total else 0
        avg_tot = total_pipeline_time / total if total else 0

        logger.info(f"Total images: {total}")
        logger.info(f"Total preprocess time: {total_preprocess_time:.4f}s, avg: {avg_pre:.6f}s")
        logger.info(f"Total inference time: {total_infer_time:.4f}s, avg: {avg_inf:.6f}s")
        logger.info(f"Total postprocess time: {total_postprocess_time:.4f}s, avg: {avg_post:.6f}s")
        logger.info(f"Total pipeline time: {total_pipeline_time:.4f}s, avg: {avg_tot:.6f}s")

        return coco_results

    def eval(self, pred_results, anno_json_path):
        """
        使用 COCO API 评估 mAP（bbox + segmentation）

        Args:
            pred_results: list, COCO 格式检测结果
            anno_json_path: str, COCO 格式标注文件路径

        Returns:
            metrics: dict, 评估指标
        """
        assert os.path.isfile(anno_json_path), f"Annotation file not found: {anno_json_path}"

        pred_json_path = "predictions.json"
        logger.info(f'Saving predictions to {pred_json_path}...')
        with open(pred_json_path, 'w') as f:
            json.dump(pred_results, f)

        coco_gt = COCO(anno_json_path)
        coco_dt = coco_gt.loadRes(pred_json_path)

        metrics = {}
        for iou_type in ['bbox', 'segm']:
            logger.info(f'\nEvaluating {iou_type} mAP...')
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            metrics[f"{iou_type}_map"] = float(coco_eval.stats[0])
            metrics[f"{iou_type}_map50"] = float(coco_eval.stats[1])
            metrics[f"{iou_type}_map75"] = float(coco_eval.stats[2])
            metrics[f"{iou_type}_map_small"] = float(coco_eval.stats[3])
            metrics[f"{iou_type}_map_medium"] = float(coco_eval.stats[4])
            metrics[f"{iou_type}_map_large"] = float(coco_eval.stats[5])

        return metrics
