# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/23 15:44
@File    : bench_yolov5_triton_npy.py
@Author  : zj
@Description: 
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

from typing import List
from core.backends.triton_model import TritonModel


class TritonDetectModel(TritonModel):
    """检测模型"""

    def __init__(
            self,
            model_name: str,
            class_list: List[str],
            half: bool = False,
            server_url: str = 'localhost:8001',
            protocol: str = 'grpc',
    ):
        super().__init__(
            model_name=model_name,
            class_list=class_list,
            half=half,
            server_url=server_url,
            protocol=protocol,
        )


"""
 Detect_COCO_YOLOv5s_TensorRT

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.558
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.157
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.354
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.651
"""

if __name__ == '__main__':
    with Profile(name="evaluating") as stage_profiler:
        input_size = 640
        # model_name = "Detect_COCO_YOLOv5s_ONNX"
        model_name = "Detect_COCO_YOLOv5s_TensorRT"
        from core.cfgs.coco_cfg import class_list

        model = TritonDetectModel(model_name, class_list)
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
