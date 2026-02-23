# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/23 14:52
@File    : bench_yolov8seg_tensorrt_npy.py
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
from core.npy.yolov8seg_evaluator import EvalEvaluator
from core.npy.yolov8_preprocess import ImgPrepare
from core.npy.segment_dataset import SegmentDataset
from core.utils.helpers import Profile
from core.utils.logger import EnhancedLogger, LOGGER_NAME

import logging

logger = EnhancedLogger(LOGGER_NAME, log_dir='logs',
                        log_file=f'{LOGGER_NAME}.log', level=logging.INFO, backup_count=30,
                        use_file_handler=True, use_stream_handler=True).logger

import numpy as np
from core.backends.trt_model import TRTModel, List, Union, Tuple


class TRTSegmentModel(TRTModel):
    """分割模型（继承自统一基类）"""

    def __init__(
            self,
            model_path: str,
            class_list: List[str],
            half: bool = False,
            stride: int = 32,
    ):
        super().__init__(
            engine_path=model_path,
            class_list=class_list,
            half=half,
            stride=stride,
        )

"""
Evaluating bbox mAP...
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=21.52s).
Accumulating evaluation results...
DONE (t=3.68s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.42731
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.59800
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.46380
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.21865
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.46630
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.60133
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.33860
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.54772
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.59004
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.36449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.64606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.75319
Evaluating segm mAP...
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=21.91s).
Accumulating evaluation results...
DONE (t=4.04s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.33529
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.55801
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.34525
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.12247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.37301
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.52034
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.28051
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.43077
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.45555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.20825
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.51496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.65666
"""

if __name__ == '__main__':
    with Profile(name="evaluating") as stage_profiler:
        input_size = 640
        model_path = "./models/tensorrt/yolov8s-seg_fp16.engine"
        from core.cfgs.coco_cfg import class_list

        model = TRTSegmentModel(model_path, class_list)
        data_root = "/workdir/datasets/coco-seg"
        dataset = SegmentDataset(data_root, class_list, img_size=input_size)
        transform = ImgPrepare(input_size, half=False)

        # conf_thres = 0.25
        conf_thres = 0.001
        iou_thres = 0.7
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
