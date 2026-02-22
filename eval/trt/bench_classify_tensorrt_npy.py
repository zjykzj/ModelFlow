# -*- coding: utf-8 -*-

"""
@Time    : 2026/2/22 19:39
@File    : bench_classify_tensorrt_npy.py
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

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA 上下文（仅用于脚本模式）


class TRTClassifyModel:
    def __init__(self, engine_path, class_list, label_list, input_size=224, fp16=False):
        self.engine_path = engine_path
        self.class_list = class_list
        self.label_list = label_list
        self.input_size = input_size
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

        logger.info(f"Model Input Names: {self.input_names}")
        logger.info(f"Model Input Shapes: {self.input_shapes}")
        logger.info(f"Model Output Names: {self.output_names}")
        logger.info(f"Model Output Shapes: {self.output_shapes}")

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
        # 假设输入输出都是 float32（常见于分类模型）
        self.input_nbytes = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_nbytes = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize

        # 分配 GPU 显存
        self.d_input = cuda.mem_alloc(self.input_nbytes)
        self.d_output = cuda.mem_alloc(self.output_nbytes)

        # 主机内存（用于拷贝）
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

    def __warmup(self):
        dummy = np.random.randn(*self.input_shape).astype(np.float32)
        for _ in range(3):
            self.infer(dummy)

    def infer(self, img: np.ndarray):
        assert img.shape == self.input_shape, f"Input shape mismatch: {img.shape} vs {self.input_shape}"
        # 将数据复制到 pinned memory
        np.copyto(self.h_input, img.ravel())

        # 创建 CUDA 流
        stream = cuda.Stream()
        # H2D
        cuda.memcpy_htod_async(self.d_input, self.h_input, stream)
        # 推理
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=stream.handle
        )
        # D2H
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, stream)
        # 同步
        stream.synchronize()

        # reshape 输出
        return self.h_output.reshape(self.output_shape)

    def __call__(self, img):
        return self.infer(img)

    def forward(self, img):
        return self.infer(img)


if __name__ == '__main__':
    with Profile(name="evaluating") as stage_profiler:
        model_path = "./models/tensorrt/efficientnet_b0_fp16.engine"
        assert os.path.isfile(model_path), model_path
        from core.cfgs.imagenet_cfg import class_list, label_list

        input_size = 224
        model = TRTClassifyModel(model_path, class_list, label_list, input_size=input_size)

        data_root = "/workdir/datasets/imagenet/val"
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
