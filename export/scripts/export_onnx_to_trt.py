# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 16:45
@File    : export_onnx_to_trt.py
@Author  : zj
@Description: 
"""

import os
import sys
import argparse

# 将当前工作目录加入 Python 模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trt_exporter import convert_onnx_to_tensorrt
from core.trt_verify import verify_trt_output


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine")
    parser.add_argument("--onnx", type=str, required=True, help="Path to input ONNX model")
    parser.add_argument("--engine", type=str, required=True, help="Path to save TensorRT engine")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size")
    parser.add_argument("--verbose", action="store_false", help="Print detailed logs")

    args = parser.parse_args()

    # Step 1: Convert ONNX to TRT
    convert_onnx_to_tensorrt(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    # Step 2: Optional verification
    verify_trt_output(args.onnx, args.engine, batch_size=args.batch_size, is_fp16=args.fp16)


if __name__ == "__main__":
    main()
