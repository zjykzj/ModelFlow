# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 20:51
@File    : bench_yolov5_tensorrt_npy.py
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

import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA 上下文（仅用于脚本模式）


class TRTDetectModel:
    def __init__(self, engine_path, class_list, fp16=False):
        self.engine_path = engine_path
        self.class_list = class_list
        self.fp16 = fp16

        logger.info(f"Loading TensorRT engine from: {engine_path}")
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()

        # 解析 bindings
        self.input_names = []
        self.output_names = []
        self.input_shapes = []
        self.output_shapes = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            is_input = self.engine.binding_is_input(i)

            if is_input:
                self.input_names.append(name)
                self.input_shapes.append(shape)
            else:
                self.output_names.append(name)
                self.output_shapes.append(shape)

            logger.info(f"Binding[{i}]: name='{name}', shape={shape}, dtype={dtype}, is_input={is_input}")

        logger.info(f"input_names: {self.input_names} - output_names: {self.output_names}")
        logger.info(f"input_shapes: {self.input_shapes} - output_shapes: {self.output_shapes}")

        # 仅支持单输入单输出分类模型
        assert len(self.input_names) == 1, "Only single input is supported."
        assert len(self.output_names) == 1, "Only single output is supported."

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        self.input_shape = self.input_shapes[0]
        self.output_shape = self.output_shapes[0]

        self._allocate_buffers()
        self.__warmup()

    def _load_engine(self):
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        """分配 GPU 和主机内存（支持多输入多输出）"""
        self.d_inputs = []
        self.d_outputs = []
        self.h_inputs = []
        self.h_outputs = []

        # 输入缓冲区
        for i, shape in enumerate(self.input_shapes):
            nbytes = trt.volume(shape) * np.dtype(np.float32).itemsize
            self.d_inputs.append(cuda.mem_alloc(nbytes))
            self.h_inputs.append(cuda.pagelocked_empty(trt.volume(shape), dtype=np.float32))

        # 输出缓冲区
        for i, shape in enumerate(self.output_shapes):
            nbytes = trt.volume(shape) * np.dtype(np.float32).itemsize
            self.d_outputs.append(cuda.mem_alloc(nbytes))
            self.h_outputs.append(cuda.pagelocked_empty(trt.volume(shape), dtype=np.float32))

    def __warmup(self):
        """模型预热"""
        dummy_inputs = [np.random.randn(*shape).astype(np.float32) for shape in self.input_shapes]
        for _ in range(3):
            self.infer(dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs)

    def infer(self, input_data):
        """
        模型推理（与 ONNXDetectModel 返回格式一致）

        Args:
            input_data: np 数组或 list[np 数组]
                - 单输入：np 数组，形状 (1, 3, 640, 640)
                - 多输入：list[np 数组]

        Returns:
            list[np 数组]: 与 ONNX session.run() 返回格式一致
                - YOLOv5: [detections] 其中 detections shape=(1, 25200, 85)
        """
        # 处理输入（支持单输入或多输入）
        if isinstance(input_data, np.ndarray):
            input_data = [input_data]

        assert len(input_data) == len(self.input_shapes), \
            f"Input count mismatch: {len(input_data)} vs {len(self.input_shapes)}"

        # 创建 CUDA 流
        stream = cuda.Stream()

        # H2D 传输（所有输入）
        for i, (h_inp, d_inp, inp) in enumerate(zip(self.h_inputs, self.d_inputs, input_data)):
            assert inp.shape == self.input_shapes[i], \
                f"Input {i} shape mismatch: {inp.shape} vs {self.input_shapes[i]}"
            np.copyto(h_inp, inp.ravel())
            cuda.memcpy_htod_async(d_inp, h_inp, stream)

        # 构建 bindings 列表（输入 + 输出）
        bindings = [int(d) for d in self.d_inputs] + [int(d) for d in self.d_outputs]

        # 推理
        self.context.execute_async_v2(
            bindings=bindings,
            stream_handle=stream.handle
        )

        # D2H 传输（所有输出）
        for h_out, d_out in zip(self.h_outputs, self.d_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, stream)

        # 同步等待完成
        stream.synchronize()

        # reshape 输出并返回 list（与 ONNX Runtime 一致）
        outputs = [h_out.reshape(shape) for h_out, shape in zip(self.h_outputs, self.output_shapes)]
        return outputs

    def __call__(self, input_data):
        """直接调用"""
        return self.infer(input_data)

    def forward(self, input_data):
        """前向传播"""
        return self.infer(input_data)

"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.36448
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.55865
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.38843
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.15721
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.35330
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.50000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.29747
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.48515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.52462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.28797
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.53612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.65122
"""

if __name__ == '__main__':
    with Profile(name="evaluating") as stage_profiler:
        model_path = "./models/tensorrt/yolov5s_fp16.engine"
        from core.cfgs.coco_cfg import class_list

        model = TRTDetectModel(model_path, class_list)

        input_size = 640
        data_root = "/workdir/datasets/coco/"
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
