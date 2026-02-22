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
import onnxruntime as ort


class ONNXClassifyModel:

    def __init__(self, model_path, class_list, label_list, half=False, device=None, input_size=224):
        self.model_path = model_path
        self.class_list = class_list
        self.label_list = label_list
        self.half = half
        self.device = device
        self.input_size = input_size

        logger.info(f"model_path: {model_path}")
        # Force CPU execution (ignore device argument per requirement)
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        logger.info(f"session: {self.session}")
        # 获取输入信息
        inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in inputs]
        self.input_shapes = [inp.shape for inp in inputs]

        # 获取输出信息
        outputs = self.session.get_outputs()
        self.output_names = [out.name for out in outputs]
        self.output_shapes = [out.shape for out in outputs]

        logger.info(f"input_names: {self.input_names} - output_names: {self.output_names}")
        logger.info(f"input_shapes: {self.input_shapes} - output_shapes: {self.output_shapes}")

        self.__warmup()

    def __warmup(self):
        dummy_input = np.random.randn(1, 3, self.input_size, self.input_size).astype(np.float32)
        for _ in range(3):
            self.session.run(None, {self.input_names[0]: dummy_input})

    def __call__(self, img):
        outputs = self.session.run(None, {self.input_names[0]: img})
        logits = outputs[0]
        return logits

    def forward(self, img):
        return self.__call__(img)

    def infer(self, img):
        return self.__call__(img)


if __name__ == '__main__':
    with Profile(name="evaluating") as stage_profiler:
        model_path = "./models/runtime/efficientnet_b0.onnx"
        assert os.path.isfile(model_path), model_path
        from core.cfgs.imagenet_cfg import class_list, label_list

        input_size = 224
        model = ONNXClassifyModel(model_path, class_list, label_list, input_size=input_size)

        data_root = "/home/zjykzj/datasets/imagenet/val"
        assert os.path.isdir(data_root), data_root
        dataset = ClassifyDataset(data_root, class_list, label_list, imread=imread_pil)

        transform = ImgPrepare(input_size=256, crop_size=224, batch=True, mode="crop")
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
