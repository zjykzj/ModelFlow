# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 10:20
@File    : classify_evaluator.py
@Author  : zj
@Description: 
"""

import time

import numpy as np
from tqdm import tqdm

from core.utils.logger import get_logger, LOGGER_NAME

logger = get_logger(LOGGER_NAME)


def calculate_metrics(confusion_matrix, classes_list):
    """
    根据混淆矩阵计算整体准确率、精度、召回率、F1-Score，
    以及识别正确的数目和识别错误的数目，并支持逐类别统计。

    参数:
        confusion_matrix (numpy.ndarray): 混淆矩阵，形状为 (C, C)，C 是类别数。

    返回:
        dict: 包含以下内容：
            - Accuracy（整体准确率）
            - Macro-Precision（宏平均精度）
            - Macro-Recall（宏平均召回率）
            - Macro-F1-Score（宏平均F1分数）
            - PerClass_XXX（按类别分类的指标）
            - Correct Count / Error Count
    """
    confusion_matrix = np.array(confusion_matrix)
    num_classes = confusion_matrix.shape[0]

    # 初始化每类指标列表
    per_class_precision = []
    per_class_recall = []
    per_class_accuracy = []

    # 存储每个类别的指标
    class_metrics = {}

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = np.sum(confusion_matrix) - TP - FP - FN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        # 计算 F1 Score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_accuracy.append(accuracy)

        class_name = classes_list[i]  # 注意这里需要从 dataset 获取类别名
        class_metrics[class_name] = {
            # "class_name": class_name,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),  # 添加 F1 Score 字段
            "accuracy": float(accuracy)
            # "Precision": float(precision),
            # "Recall": float(recall),
            # "Accuracy": float(accuracy)
        }

    # 宏平均
    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (
                                                                                                macro_precision + macro_recall) > 0 else 0

    # 微平均
    total_TP = np.trace(confusion_matrix)
    total_FP = np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    total_FN = np.sum(confusion_matrix, axis=1) - np.diag(confusion_matrix)
    micro_precision = total_TP / (total_TP + sum(total_FP)) if (total_TP + sum(total_FP)) > 0 else 0
    micro_recall = total_TP / (total_TP + sum(total_FN)) if (total_TP + sum(total_FN)) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                micro_precision + micro_recall) > 0 else 0

    # 整体准确率
    total_samples = np.sum(confusion_matrix)
    accuracy = total_TP / total_samples if total_samples > 0 else 0

    # 正确/错误识别数量
    correct_count = total_TP
    error_count = total_samples - correct_count

    return {
        "Accuracy": float(accuracy),
        "Macro-Precision": float(macro_precision),
        "Macro-Recall": float(macro_recall),
        "Macro-F1-Score": float(macro_f1),
        "Micro-Precision": float(micro_precision),
        "Micro-Recall": float(micro_recall),
        "Micro-F1-Score": float(micro_f1),
        "Correct Count": int(correct_count),
        "Error Count": int(error_count),
        "PerClass_Metrics": class_metrics
    }


def softmax(x, axis=1):
    # 确保输入是 NumPy 数组
    x = np.asarray(x, dtype=np.float32)

    # 为数值稳定性，减去最大值（防止 exp 溢出）
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max

    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class EvalEvaluator:
    """
    输入模型、输入数据集

    1. 加载模型
    2. 加载数据集
    3. 批量进行推理
    """

    def __init__(self, model, dataset, transform, cls_thres=None):
        self.model = model
        self.dataset = dataset
        self.transform = transform
        self.cls_thres = cls_thres

        self.error_dict = None
        self.confusion_matrix = None

        self.class_list = self.model.class_list

    def run(self):
        """
        支持多种类型的模型，包括分类算法/检测算法/分割算法

        保存结果，包括
        每张图片的图片名、标注类别名、预测类别名、最大分类概率
        """
        num_classes = len(self.class_list)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        error_dict = {}

        total_preprocess_time = 0.0  # 用于累计推理总耗时（秒）
        total_infer_time = 0.0  # 用于累计推理总耗时（秒）
        total_postprocess_time = 0.0  # 用于累计推理总耗时（秒）
        total_pipeline_time = 0.0  # 用于累计推理总耗时（秒）

        total = len(self.dataset)
        for idx in tqdm(range(len(self.dataset))):
            img, img_name, img_path, cls_idx, class_name = self.dataset.__getitem__(idx)

            # 推理开始计时
            t1 = time.time()

            input_data = self.transform(img)

            t2 = time.time()

            output = self.model(input_data)
            assert isinstance(output, np.ndarray)

            t3 = time.time()
            probs = softmax(output, axis=1)

            # 针对二分类进行特殊处理
            if len(probs[0]) == 2 and self.cls_thres is not None:
                if probs[0, 0] > self.cls_thres:
                    y_pred = 0
                    y_prob = probs[0, y_pred]
                else:
                    y_pred = 1
                    y_prob = probs[0, y_pred]
            else:
                y_pred = output[0].argmax().item()
                y_prob = probs[0, y_pred]
            pred_name = self.dataset.classes_list[y_pred]
            # print(img_name, cls_idx, class_name, y_pred, pred_name, y_prob)

            t4 = time.time()

            # 推理结束计时
            preprocess_time = t2 - t1
            infer_time = t3 - t2
            postprocess_time = t4 - t3

            total_preprocess_time += preprocess_time
            total_infer_time += infer_time
            total_postprocess_time += postprocess_time
            total_pipeline_time += (t4 - t1)

            confusion_matrix[cls_idx, y_pred] += 1
            if y_pred != cls_idx:
                if class_name not in error_dict.keys():
                    error_dict[class_name] = []
                error_dict[class_name].append({
                    'pred_name': pred_name,
                    'target_name': class_name
                })

        avg_preprocess_time = total_preprocess_time / total if total > 0 else 0.0
        avg_infer_time = total_infer_time / total if total > 0 else 0.0
        avg_postprocess_time = total_postprocess_time / total if total > 0 else 0.0
        avg_total_time = total_pipeline_time / total if total > 0 else 0.0

        logger.info(
            f"Total preprocess time: {total_preprocess_time:.4f} s, Average per image: {avg_preprocess_time:.6f} s")
        logger.info(f"Total inference time: {total_infer_time:.4f} s, Average per image: {avg_infer_time:.6f} s")
        logger.info(
            f"Total postprocess time: {total_postprocess_time:.4f} s, Average per image: {avg_postprocess_time:.6f} s")
        logger.info(f"Total pipeline time: {total_pipeline_time:.4f} s, Average per image: {avg_total_time:.6f} s")

        logger.info(f"confusion_matrix: {confusion_matrix}")
        self.confusion_matrix = confusion_matrix
        self.error_dict = error_dict

    def eval(self):
        """
        统计整体结果，包括
        多少张识别正确、多少张识别错误、准确率、精度、召回率、F1_score
        """
        logger.info(f"Process calculate metrics for {len(self.dataset)} images")

        # 计算总数、正确数目、错误数目和类别总数
        total_images_num = np.sum(self.confusion_matrix)  # 总样本数
        correct_images_num = np.trace(self.confusion_matrix)  # 对角线元素之和（正确分类的样本数）
        error_images_num = total_images_num - correct_images_num  # 错误分类的样本数
        classs_num = self.confusion_matrix.shape[0]  # 类别总数

        eval_dict = calculate_metrics(self.confusion_matrix, self.dataset.classes_list)

        result_dict = {"train_type": "classification",
                       "total_images_num": str(total_images_num), "correct_images_num": str(correct_images_num),

                       "error_images_num": str(error_images_num), "classes_num": str(classs_num),
                       "accuracy": str(eval_dict['Accuracy']),

                       "precision": str(eval_dict['Macro-Precision']),
                       "recall": str(eval_dict['Macro-Recall']),
                       "f1_score": str(eval_dict['Macro-F1-Score']),

                       'per_class': {}}
        for class_name, item_dict in eval_dict['PerClass_Metrics'].items():
            error_list = []
            if class_name in self.error_dict.keys():
                error_list = self.error_dict[class_name]
            item_dict['error_data'] = error_list

            result_dict['per_class'][class_name] = item_dict

        return result_dict
