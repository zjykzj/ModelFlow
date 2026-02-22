# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 14:23
@File    : segment_evaulator.py
@Author  : zj
@Description: 
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import json
import time
import cv2

from typing import Optional, Union, List, Tuple

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.logger import get_logger

logger = get_logger()


def xywh2xyxy_numpy(xywh: np.ndarray) -> np.ndarray:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    xyxy = np.copy(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
    return xyxy


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Pure numpy NMS."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    sorted_indices = np.argsort(scores)[::-1]
    keep = []
    while len(sorted_indices) > 0:
        i = sorted_indices[0]
        keep.append(i)
        if len(sorted_indices) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[sorted_indices[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[sorted_indices[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[sorted_indices[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[sorted_indices[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[sorted_indices[1:], 2] - boxes[sorted_indices[1:], 0]) * \
                      (boxes[sorted_indices[1:], 3] - boxes[sorted_indices[1:], 1])
        union = area_i + area_others - inter
        iou = inter / (union + 1e-6)
        keep_indices = np.where(iou <= iou_threshold)[0]
        sorted_indices = sorted_indices[1:][keep_indices]
    return np.array(keep, dtype=np.int64)


def non_max_suppression_seg(
        predictions,
        conf_thres=0.25,
        iou_thres=0.45,
        classes: Optional[Union[List[int], np.ndarray]] = None,
        agnostic=False,
        multi_label=False,
        max_det=300
):
    """
    Runs Non-Maximum Suppression (NMS) on inference results (NumPy version).
    predictions: list of [pred, _, mask_coeffs]
        - pred: (bs, 8400, 8) -> [xywh, obj, cls]
        - mask_coeffs: (bs, 8400, 33) -> mask coefficients (assume 33 = 32 proto + 1 bias?)
    """
    prediction = predictions[0]  # (bs, 8400, 8)
    confs = predictions[2]  # (bs, 8400, 33), your 'output2_Transpose_f32'

    # 拼接 detection 和 mask coefficients
    prediction = np.concatenate([prediction, confs], axis=2)  # (bs, 8400, 8+33=41)

    # 注意：原公式 num_classes = prediction.shape[2] - 5 - 33 = 41 - 5 - 33 = 3 ✅
    # ✅ 自动推断 num_classes
    num_classes = prediction.shape[2] - 5 - 33
    # print(f"[NMS] Detected num_classes = {num_classes} (from prediction.shape: {prediction.shape})")

    # 筛选候选框: obj_conf > conf_thres 且 max(cls_conf) > conf_thres
    obj_conf = prediction[..., 4]  # (bs, 8400)
    cls_conf = prediction[..., 5:5 + num_classes]  # (bs, 8400, 3)
    max_cls_conf = np.max(cls_conf, axis=-1)  # (bs, 8400)
    pred_candidates = (obj_conf > conf_thres) & (max_cls_conf > conf_thres)  # (bs, 8400)

    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    max_wh = 4096
    max_nms = 30000
    time_limit = 10.0
    multi_label = multi_label and (num_classes > 1)

    tik = time.time()
    bs = prediction.shape[0]
    output = [np.zeros((0, 6 + 33), dtype=np.float32) for _ in range(bs)]  # (x1y1x2y2, conf, cls, mask_coeffs[33])

    for img_idx in range(bs):
        x = prediction[img_idx]  # (8400, 41)
        x = x[pred_candidates[img_idx]]  # 筛选

        if x.shape[0] == 0:
            continue

        # conf = obj_conf * cls_conf
        x[:, 5:5 + num_classes] *= x[:, 4:5]

        # xywh -> xyxy
        box = xywh2xyxy_numpy(x[:, :4])  # (N, 4)
        segconf = x[:, 5 + num_classes:]  # (N, 33), from 'confs'

        if multi_label:
            # 多标签：所有 cls_conf > conf_thres 的都保留
            cls_mask = x[:, 5:5 + num_classes] > conf_thres  # (N, 3)
            box_idx, class_idx = np.where(cls_mask)  # (k,), (k,)
            detections = np.concatenate([
                box[box_idx],
                x[box_idx, class_idx + 5].reshape(-1, 1),  # conf
                class_idx.reshape(-1, 1).astype(np.float32),
                segconf[box_idx]
            ], axis=1)
        else:
            # 单标签：取最高分的类
            conf = np.max(x[:, 5:5 + num_classes], axis=1)  # (N,)
            class_idx = np.argmax(x[:, 5:5 + num_classes], axis=1)  # (N,)
            x = np.concatenate([box, conf.reshape(-1, 1), class_idx.reshape(-1, 1).astype(np.float32), segconf], axis=1)
            x = x[conf > conf_thres]  # 过滤
            detections = x

        # Filter by classes
        if classes is not None:
            classes = np.array(classes)
            keep = np.isin(detections[:, 5], classes)
            detections = detections[keep]
            if len(detections) == 0:
                continue

        # Limit to max_nms
        if detections.shape[0] > max_nms:
            idx = np.argsort(detections[:, 4])[::-1][:max_nms]
            detections = detections[idx]

        # Class-agnostic NMS
        class_offset = detections[:, 5:6] * (0 if agnostic else max_wh)
        boxes_nms = detections[:, :4] + class_offset
        scores_nms = detections[:, 4]
        keep_box_idx = nms_numpy(boxes_nms, scores_nms, iou_thres)

        if keep_box_idx.shape[0] > max_det:
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = detections[keep_box_idx]

        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break

    return output


def crop_mask_numpy(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.

    Args:
        masks: (n, h, w) float32, mask logits or sigmoid outputs
        boxes: (n, 4) float32, in *relative* coordinates [x1, y1, x2, y2] in range [0, 1]

    Returns:
        cropped masks: (n, h, w), values outside bbox are 0
    """
    n, h, w = masks.shape

    # Expand boxes to (n, 1, 1) for broadcasting
    x1 = boxes[:, 0:1, None]  # (n, 1, 1)
    y1 = boxes[:, 1:2, None]
    x2 = boxes[:, 2:3, None]
    y2 = boxes[:, 3:4, None]

    # Grid coordinates in [0,1] space
    r = np.linspace(0, 1, w, dtype=np.float32)[None, None, :]  # (1, 1, w)
    c = np.linspace(0, 1, h, dtype=np.float32)[None, :, None]  # (1, h, 1)

    # Create binary mask: inside bbox
    inside_box = ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))  # (n, h, w)

    return masks * inside_box.astype(masks.dtype)


def handle_proto_test_numpy(
        proto_list: List[np.ndarray],
        oconfs: np.ndarray,
        imgshape: Tuple[int, int],
        img_orishape: Optional[Tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """
    Reconstruct masks from prototypes and coefficients.
    Now supports oconfs as (N, 39) directly (without batch dim).
    """
    # ==================== 处理输入维度 ====================
    # oconfs 可能是 (N, 39) 而不是 (bs, N, 39)
    if oconfs.ndim == 2:
        # 假设 batch_size = 1
        oconfs = oconfs[None, ...]  # -> (1, N, 39)
    bs, num_boxes = oconfs.shape[:2]

    if num_boxes == 0:
        return None

    conf = oconfs[..., 6:]  # (1, N, 33)
    xyxy = oconfs[..., :4]  # (1, N, 4)
    confs = conf[..., 1:]  # (1, N, 32)

    proto = proto_list[0]  # (1, 32, ph, pw)
    _, _, ph, pw = proto.shape

    # Normalize xyxy to [0,1]
    xyxy_rel = xyxy / 640.0  # 假设输入是 640x640

    # Reshape proto: (1, 32, ph*pw)
    proto_reshaped = proto.reshape(1, 32, -1)  # (1, 32, ph*pw)

    # 生成 mask: (1, N, 32) @ (1, 32, ph*pw) -> (1, N, ph*pw)
    seg = np.matmul(confs, proto_reshaped)  # batch matmul
    seg = seg.reshape(1, num_boxes, ph, pw)  # (1, N, 160, 160)

    # Sigmoid
    seg = 1.0 / (1.0 + np.exp(-seg))

    # Resize all masks
    masks_list = []
    for b in range(1):  # bs=1
        masks_b = []
        for i in range(num_boxes):
            mask = seg[b, i]  # (ph, pw)
            resized = cv2.resize(mask, (imgshape[1], imgshape[0]),
                                 interpolation=cv2.INTER_LINEAR)
            masks_b.append(resized)
        masks_list.append(np.stack(masks_b, axis=0))  # (N, H, W)

    masks = masks_list[0]  # (N, H, W)

    # 裁剪 mask
    masks_cropped = crop_mask_numpy(masks, xyxy_rel[0])  # (N, H, W)

    # 二值化
    masks_binary = (masks_cropped > 0.5)  # bool array

    return masks_binary  # (N, H, W), bool


def rescale_box(ori_shape, boxes, target_shape, ratio, padding):
    """
    Rescale the output to the original image shape.
    Args:
        ori_shape: Original shape (height, width) of the processed image, e.g., (640, 640).
        boxes: Detected bounding boxes in format [x1, y1, x2, y2].
        target_shape: Target shape (height, width) of the original image, e.g., (200, 200).
        ratio: The scale ratio used during preprocessing.
        padding: Padding applied during preprocessing in format (left, top).
    Returns:
        Rescaled bounding boxes.
    """
    # Adjust for padding and scale back to original size
    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    # Clip boxes to image boundaries
    boxes[:, 0].clip(0, target_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, target_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, target_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, target_shape[0], out=boxes[:, 3])  # y2

    return boxes.round().astype(int)


def rescale_mask(masks, target_shape, ratio, padding):
    """
    Rescale the mask output to the original image shape.

    Args:
        masks: Masks in format (N, H, W).
        target_shape: Target shape (height, width) of the original image.
        ratio: The scale ratio used during preprocessing.
        padding: Padding applied during preprocessing in format (left, top).

    Returns:
        Rescaled masks.
    """
    # Crop padding from masks
    masks = masks[:, int(padding[1]):int(masks.shape[1] - padding[1]), int(padding[0]):int(masks.shape[2] - padding[0])]

    # Resize masks to target shape using nearest neighbor interpolation
    resized_masks = []
    for mask in masks:
        resized_mask = cv2.resize(mask.astype(np.uint8), tuple(target_shape[::-1]), interpolation=cv2.INTER_NEAREST)
        resized_masks.append(resized_mask)

    return np.array(resized_masks, dtype=bool)


###################################################### DRAW

def generate_colors(i, bgr=False):
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
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    common_color = [[128, 0, 0], [255, 0, 0], [255, 0, 255], [255, 102, 0], [51, 51, 0], [0, 51, 0], [51, 204, 204],
                    [0, 128, 128], [0, 204, 255]]
    if segment is not None:
        import random
        # ii=random.randint(0, len(common_color)-1)
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
    def __init__(self, model, dataset, transform, conf_thres=0.4, iou_thres=0.45, max_det=300):
        self.model = model
        self.dataset = dataset
        self.transform = transform
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.class_list = model.class_list

    def run(self, save=False):
        total_preprocess_time = 0.0
        total_infer_time = 0.0
        total_postprocess_time = 0.0
        total_pipeline_time = 0.0
        coco_results = []

        total = len(self.dataset)
        for idx in tqdm(range(total), desc="Evaluating"):
            img_src, path, _ = self.dataset[idx]

            t1 = time.time()
            data, ratio, padding = self.transform(img_src)
            t2 = time.time()

            pred_results = self.model(data)
            t3 = time.time()

            loutputs = non_max_suppression_seg(
                pred_results,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                agnostic=False,
                max_det=self.max_det
            )

            # 提取检测结果与原型掩码
            protos = pred_results[1]  # 原型掩码 [1, 32, 160, 160]
            segconf = [loutputs[li][..., 0:] for li in range(len(loutputs))]
            det = [loutputs[li][..., :6] for li in range(len(loutputs))][0]  # 第一个 batch

            # 生成实例分割掩码
            segments = \
                [handle_proto_test_numpy([protos[li].reshape(1, *(protos[li].shape[-3:]))], segconf[li],
                                         data.shape[2:])
                 for li in range(len(loutputs))][0]

            img_ori = img_src.copy()
            if len(det):
                det[:, :4] = rescale_box(data.shape[2:],
                                         det[:, :4],
                                         img_src.shape[:2],
                                         ratio,
                                         padding
                                         )

                boxes_xyxy = det[:, :4]
                scores = det[:, 4]
                classes = det[:, 5]

                segments = rescale_mask(
                    segments,
                    img_src.shape[:2],
                    ratio,
                    padding
                )

            t4 = time.time()
            total_preprocess_time += t2 - t1
            total_infer_time += t3 - t2
            total_postprocess_time += t4 - t3
            total_pipeline_time += t4 - t1

            if len(det):
                if save:
                    ii = len(det) - 1
                    for *xyxy, conf, cls in reversed(det):
                        class_num = int(cls)  # integer class
                        label = f'{self.class_list[class_num]} {conf:.2f}'

                        img_ori = plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy,
                                                     label,
                                                     color=generate_colors(class_num, True), segment=segments[ii])
                        ii -= 1
                    img_src = np.asarray(img_ori)

                # Ensure masks are uint8 and binary (0 or 1)
                masks = (segments > 0.5).astype(np.uint8)  # 关键：转为 uint8 二值掩码
                # Encode masks as RLE
                if masks.size > 0:
                    masks_fortran = np.asfortranarray(masks.transpose(1, 2, 0))  # [H, W, N]
                    rles = maskUtils.encode(masks_fortran)
                    if isinstance(rles, dict):  # single mask case
                        rles = [rles]
                    for rle in rles:
                        rle['counts'] = rle['counts'].decode('utf-8')
                else:
                    rles = []

                # Append predictions
                for i in range(len(det)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    bbox_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    rle = rles[i] if rles else []

                    coco_results.append({
                        "image_id": int(Path(path).stem),
                        "category_id": int(classes[i].item()) + 1,  # COCO class starts at 1
                        "bbox": bbox_xywh,
                        "score": float(scores[i].item()),
                        "segmentation": rle
                    })

            if save:
                cv2.imwrite(f"seg-{Path(path).stem}.jpg", img_src)

        # Logging
        avg_pre = total_preprocess_time / total if total else 0
        avg_inf = total_infer_time / total if total else 0
        avg_post = total_postprocess_time / total if total else 0
        avg_tot = total_pipeline_time / total if total else 0

        logger.info(f"Total preprocess time: {total_preprocess_time:.4f}s, avg: {avg_pre:.6f}s")
        logger.info(f"Total inference time: {total_infer_time:.4f}s, avg: {avg_inf:.6f}s")
        logger.info(f"Total postprocess time: {total_postprocess_time:.4f}s, avg: {avg_post:.6f}s")
        logger.info(f"Total pipeline time: {total_pipeline_time:.4f}s, avg: {avg_tot:.6f}s")

        return coco_results

    def eval(self, pred_results, anno_json_path):
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
            metrics[f"{iou_type}_map"] = coco_eval.stats[0]
            metrics[f"{iou_type}_map50"] = coco_eval.stats[1]

        return metrics
