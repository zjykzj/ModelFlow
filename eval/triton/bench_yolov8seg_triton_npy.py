# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/23 16:50
@File    : bench_yolov8seg_triton_npy.py
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

from eval.triton.bench_yolov5_triton_npy import TritonDetectModel


class TritonSegmentModel(TritonDetectModel):
    pass

"""
Evaluating bbox mAP...
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=18.66s).
Accumulating evaluation results...
DONE (t=3.30s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.598
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.753
Evaluating segm mAP...
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=19.53s).
Accumulating evaluation results...
DONE (t=3.73s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.336
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.559
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.346
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.281
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.656
"""

if __name__ == '__main__':
    with Profile(name="evaluating") as stage_profiler:
        input_size = 640
        # model_name = "Segment_COCO_YOLOv8sSeg_ONNX"
        model_name = "Segment_COCO_YOLOv8sSeg_TensorRT"
        from core.cfgs.coco_cfg import class_list

        model = TritonSegmentModel(model_name, class_list)
        data_root = "/home/zjykzj/datasets/coco-seg"
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
