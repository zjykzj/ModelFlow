# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 17:39
@File    : main.py
@Author  : zj
@Description:

>>>python3 py/runtime/bench_classify_onnx_npy.py

"""

import sys
import os

# ==================== 配置项目根目录 ====================
# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 当前脚本路径: ~/ModelFlow/py/runtime/
# 需要回到项目根目录: ~/ModelFlow/
project_root = os.path.dirname(os.path.dirname(current_dir))  # 向上两级

# 将项目根目录加入 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[INFO] Added to sys.path: {project_root}")

# ==================== 然后再导入你的模块 ====================
from core.npy.classify_evaluator import EvalEvaluator
from core.npy.classify_preprocess import ImgPrepare
from core.npy.classify_dataset import ClassifyDataset, imread_pil
from core.utils.helpers import Profile
from core.utils.logger import EnhancedLogger, LOGGER_NAME

import logging

logger = EnhancedLogger(LOGGER_NAME, log_dir='logs',
                        log_file=f'{LOGGER_NAME}.log', level=logging.INFO, backup_count=30,
                        use_file_handler=True, use_stream_handler=True).logger

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from core.backends.onnx_model import ONNXModel


class ONNXClassifyModel(ONNXModel):
    """分类模型（继承自统一基类）"""

    def __init__(
            self,
            model_path: str,
            class_list: List[str],
            label_list: Optional[List[str]] = None,
            half: bool = False,
            device: Optional[str] = None,
    ):
        super().__init__(
            model_path=model_path,
            class_list=class_list,
            label_list=label_list,
            half=half,
            device=device,
        )

    def predict(self, img: np.ndarray) -> Tuple[int, float]:
        """
        分类预测，返回类别索引和置信度

        Args:
            img: 预处理后的图像数组 (1, C, H, W)

        Returns:
            (class_index, confidence)
        """
        logits = self.__call__(img)
        probs = self._softmax(logits)
        class_idx = int(np.argmax(probs[0]))
        confidence = float(probs[0][class_idx])
        return class_idx, confidence

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


"""
Evaluation Summary:
  Task Type: classification
  Total Images: 50000
  Total Correct Predictions: 38839
  Total Errors: 11161
  Number of Classes: 1000
  Accuracy: 0.7768
  Precision: 0.7798
  Recall: 0.7768
  F1-Score: 0.7783
"""

if __name__ == '__main__':
    with Profile(name="evaluating") as stage_profiler:
        input_size = 224
        model_path = "./models/runtime/efficientnet_b0.onnx"
        assert os.path.isfile(model_path), model_path
        from core.cfgs.imagenet_cfg import class_list, label_list

        model = ONNXClassifyModel(model_path, class_list, label_list)

        data_root = "/home/zjykzj/datasets/imagenet/val"
        assert os.path.isdir(data_root), data_root
        dataset = ClassifyDataset(data_root, class_list, label_list, imread=imread_pil)

        transform = ImgPrepare(input_size=256, crop_size=input_size, batch=True, mode="crop")
        engine = EvalEvaluator(model, dataset, transform, cls_thres=None)
        print(engine)

        print(f"*" * 100)
        engine.run()
        print(f"*" * 100)
        eval_dict = engine.eval()
        logger.info(f"eval_dict keys(): {eval_dict.keys()}")

    logger.info("Evaluation Summary:")
    logger.info(f"  Task Type: {eval_dict['train_type']}")
    logger.info(f"  Total Images: {eval_dict['total_images_num']}")
    logger.info(f"  Total Correct Predictions: {eval_dict['correct_images_num']}")
    logger.info(f"  Total Errors: {eval_dict['error_images_num']}")
    logger.info(f"  Number of Classes: {eval_dict['classes_num']}")
    logger.info(f"  Accuracy: {float(eval_dict['accuracy']):.4f}")
    logger.info(f"  Precision: {float(eval_dict['precision']):.4f}")
    logger.info(f"  Recall: {float(eval_dict['recall']):.4f}")
    logger.info(f"  F1-Score: {float(eval_dict['f1_score']):.4f}")

    logger.info(f"Stage {stage_profiler.name} took {stage_profiler.dt:.2f} seconds")
