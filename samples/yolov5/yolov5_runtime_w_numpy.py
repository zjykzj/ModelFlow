# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/29 17:38
@File    : yolov5_runtime_w_numpy.py
@Author  : zj
@Description: 
"""

import time
import logging
from numpy import ndarray

from core.backends.backend_runtime import BackendRuntime
from numpy_util import preprocess, postprocess


class YOLOv5Runtime:

    def __init__(self, weight: str = 'yolov5s.onnx'):
        super().__init__()
        self.session = BackendRuntime(weight)
        self.session.load()

        input_name = self.session.get_input_names()[0]
        self.net_h, self.net_w = self.session.get_input_shapes()[input_name][2:]

    def infer(self, im: ndarray) -> ndarray:
        input_name = self.session.input_names[0]
        output_dict = self.session.infer({input_name: im})
        output_name = self.session.output_names[0]
        return output_dict[output_name]

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
