# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : segment.py
@Author  : zj
@Description: 实例分割评估器

编排 Pipeline + Dataset，收集 COCO 格式预测，委托 DataFlow-CV 计算 mAP。

用法:
    evaluator = SegmentEvaluator(pipeline, dataset, gt_json="annotations.json")
    results = evaluator.run()
"""

import os
import json
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from modelflow.core.interfaces import BaseEvaluator
from modelflow.core.registry import EVALUATORS
from modelflow.utils.logger import get_logger
from modelflow.processors.detect.ops import xywh2xyxy

logger = get_logger("modelflow.evaluator.segment")


@EVALUATORS.register("segment")
class SegmentEvaluator(BaseEvaluator):
    """实例分割评估器

    Args:
        pipeline: InferencePipeline 实例（segment 类型）
        dataset: BaseDataset 实例
        gt_json: ground truth JSON 路径
    """

    def __init__(self, pipeline, dataset, metrics=None, config=None,
                 gt_json: Optional[str] = None):
        super().__init__(pipeline, dataset, metrics, config)
        self.gt_json = gt_json or (
            dataset.get_gt_json() if hasattr(dataset, "get_gt_json") else ""
        )

    def _run_inference(self) -> list:
        predictions = []

        for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
            image, gt = self.dataset[idx]
            result = self.pipeline(image, conf_thres=0.001, iou_thres=0.5)

            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            class_ids = result.get("class_ids", [])
            masks = result.get("masks")

            h, w = image.shape[:2]
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                pred = {
                    "image_id": int(gt.get("image_id", 0)),
                    "category_id": int(class_ids[i]) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[i]),
                }
                # 添加 mask 为 RLE 编码（如有）
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    pred["segmentation"] = {"size": [h, w], "counts": mask.tolist()}
                predictions.append(pred)

        return predictions

    def run(self, save_pred_json: Optional[str] = None) -> Dict[str, float]:
        predictions = self._run_inference()

        if save_pred_json:
            os.makedirs(os.path.dirname(save_pred_json) or ".", exist_ok=True)
            with open(save_pred_json, "w") as f:
                json.dump(predictions, f)
            logger.info(f"Predictions saved to {save_pred_json}")

        if self.gt_json:
            try:
                from dataflow.evaluate import DetectionEvaluator
                df_eval = DetectionEvaluator(verbose=True)
                result = df_eval.evaluate(self.gt_json, save_pred_json or predictions)
                return {k: float(v) for k, v in result.items()}
            except ImportError:
                logger.warning("DataFlow-CV not installed. Skipping mAP calculation.")
                return {"num_predictions": len(predictions)}
        else:
            logger.warning("No gt_json provided.")
            return {"num_predictions": len(predictions)}
