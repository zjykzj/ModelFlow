# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 16:38
@File    : bench_yolov5_onnx_npy.py
@Author  : zj
@Description:

>>>python3 py/runtime/bench_yolov5_onnx_npy.py

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
from core.npy.yolov5_evaluator import EvalEvaluator
from core.npy.yolov8_preprocess import ImgPrepare
from core.npy.detect_dataset import DetectDataset
from core.utils.helpers import Profile
from core.utils.logger import EnhancedLogger, LOGGER_NAME

import logging

logger = EnhancedLogger(LOGGER_NAME, log_dir='logs',
                        log_file=f'{LOGGER_NAME}.log', level=logging.INFO, backup_count=30,
                        use_file_handler=True, use_stream_handler=True).logger

import numpy as np
import onnxruntime as ort


class ONNXDetectModel:

    def __init__(self, model_path, class_list, half=False):
        self.model_path = model_path
        self.class_list = class_list
        self.half = half

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
        dummy_inputs = [np.random.randn(*reshape).astype(np.float32) for reshape in self.input_shapes]
        for _ in range(3):
            feed_data = dict(zip(self.input_names, dummy_inputs))
            self.session.run(None, feed_data)

    def __call__(self, input_data):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_data})
        return outputs

    def forward(self, input_data):
        return self.__call__(input_data)

    def infer(self, input_data):
        return self.__call__(input_data)

"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.559
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.390
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.158
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.289
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.537
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.651
"""

if __name__ == '__main__':
    with Profile(name="evaluating") as stage_profiler:
        model_path = "./models/runtime/yolov5s.onnx"
        from core.cfgs.coco_cfg import class_list

        model = ONNXDetectModel(model_path, class_list)

        input_size = 640
        data_root = "/home/zjykzj/datasets/coco/"
        dataset = DetectDataset(data_root, class_list, img_size=input_size)

        transform = ImgPrepare(input_size, half=False)

        # conf_thres = 0.25
        # iou_thres = 0.45
        conf_thres = 0.001
        iou_thres = 0.6
        engine = EvalEvaluator(model, dataset, transform, conf_thres=conf_thres, iou_thres=iou_thres)
        print(engine)

        print(f"*" * 100)
        # pred_results = engine.run(save=True)
        pred_results = engine.run(save=False)
        print(f"*" * 100)
        anno_json_path = "./annotations.json"
        engine.dataset.get_anno_json(anno_json_path)
        engine.eval(pred_results, anno_json_path)

    logger.info(f"Stage {stage_profiler.name} took {stage_profiler.dt:.2f} seconds")
