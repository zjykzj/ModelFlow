# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/15
@File    : detect.py
@Author  : zj
@Description: 检测评估器

编排 Pipeline + Dataset，收集 COCO 格式预测，委托 DataFlow-CV 计算 mAP。

用法:
    evaluator = DetectEvaluator(pipeline, dataset, gt_json="annotations.json")
    results = evaluator.run()
    # {"mAP": 0.5, "AP50": 0.7, ...}
"""

import os
import json
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from eval.interfaces import BaseEvaluator
from utils.logger import get_logger
from utils.ops import xywh2xyxy

logger = get_logger("eval.evaluator.detect")


class DetectEvaluator(BaseEvaluator):
    """检测评估器

    Args:
        pipeline: InferencePipeline 实例
        dataset: 数据集实例
        metrics: 不用传（detection 使用 DataFlow-CV）
        config: 评估配置
        gt_json: ground truth JSON 路径（DataFlow-CV 需要）
    """

    def __init__(self, pipeline, dataset, metrics=None, config=None,
                 gt_json: Optional[str] = None):
        super().__init__(pipeline, dataset, metrics, config)
        self.gt_json = gt_json or (dataset.get_gt_json() if hasattr(dataset, "get_gt_json") else "")

    def _run_inference(self) -> list:
        """遍历数据集推理，收集 COCO 格式预测"""
        predictions = []

        for idx in tqdm(range(len(self.dataset)), desc="Evaluating"):
            image, gt = self.dataset[idx]
            tensor = self.pipeline.preprocessor(image)
            raw = self.pipeline.backend(tensor)
            # 简单后处理（不依赖 postprocessor 的 original_shape）
            pred = raw[0]
            if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
                pred = pred.transpose(0, 2, 1)
            pred = pred[0]

            nc = pred.shape[1] - 4
            boxes = pred[:, :4]
            scores = pred[:, 4:] if nc > 1 else pred[:, 4]
            if scores.ndim == 2:
                max_scores = scores.max(axis=1)
                class_ids = scores.argmax(axis=1)
            else:
                max_scores = scores
                class_ids = np.zeros_like(scores, dtype=int)

            boxes = xywh2xyxy(boxes)

            h, w = image.shape[:2]
            for i in range(len(boxes)):
                if max_scores[i] < 0.001:
                    continue
                x1, y1, x2, y2 = boxes[i].tolist()
                predictions.append({
                    "image_id": int(gt.get("image_id", 0)),
                    "category_id": int(class_ids[i]) + 1,  # COCO category_id is 1-indexed
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # xywh for COCO
                    "score": float(max_scores[i]),
                })

        return predictions

    def run(self, save_pred_json: Optional[str] = None) -> Dict[str, float]:
        """运行评估

        Args:
            save_pred_json: 保存预测结果的路径（可选）

        Returns:
            metrics dict
        """
        # 1. 推理收集
        predictions = self._run_inference()

        # 2. 保存预测 JSON
        if save_pred_json:
            os.makedirs(os.path.dirname(save_pred_json) or ".", exist_ok=True)
            with open(save_pred_json, "w") as f:
                json.dump(predictions, f)
            logger.info(f"Predictions saved to {save_pred_json}")

        # 3. 委托 DataFlow-CV 计算 mAP
        if self.gt_json:
            try:
                from dataflow.evaluate import DetectionEvaluator
                from dataflow.util.logging import LogConfig

                df_eval = DetectionEvaluator(
                    log_config=LogConfig(name="eval", verbose=True)
                )
                result = df_eval.evaluate(self.gt_json, save_pred_json or predictions)
                return self._to_metrics(result)
            except ImportError:
                logger.warning("DataFlow-CV not installed. Returning raw predictions count.")
                return {"num_predictions": len(predictions)}
        else:
            logger.warning("No gt_json provided. Returning raw predictions count.")
            return {"num_predictions": len(predictions)}

    @staticmethod
    def _to_metrics(df_result) -> Dict[str, float]:
        """将 DataFlow-CV EvaluationResult 转为标准 metrics dict"""
        if not getattr(df_result, "success", False) or df_result.metrics is None:
            return {}
        m = df_result.metrics
        return {
            "mAP": m.ap,
            "AP50": m.ap50,
            "AP75": m.ap75,
            "AP_small": m.ap_small,
            "AP_medium": m.ap_medium,
            "AP_large": m.ap_large,
            "AR_max_1": m.ar_max_1,
            "AR_max_10": m.ar_max_10,
            "AR_max_100": m.ar_max_100,
            "AR_small": m.ar_small,
            "AR_medium": m.ar_medium,
            "AR_large": m.ar_large,
        }
