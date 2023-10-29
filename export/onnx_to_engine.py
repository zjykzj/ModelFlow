# -*- coding: utf-8 -*-

"""
@Time    : 2023/10/29 15:55
@File    : onnx_to_engine.py
@Author  : zj
@Description: Refer to https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/onnx_helper.py
"""

import argparse

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit


def parse_opt():
    parser = argparse.ArgumentParser(description="ONNX to Engine")
    parser.add_argument("onnx", metavar="ONNX", type=str,
                        help="ONNX Model path, default: None")
    parser.add_argument("engine", metavar="ENGINE", type=str,
                        help="Saving engine path, default: None")
    parser.add_argument("--batch-size", metavar="BATCH", type=int, default=1,
                        help="Batch size, default: 1")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Is the batch dimension dynamically set, default: False")

    args = parser.parse_args()
    print(f"args: {args}")

    return args


def convert_onnx_to_engine(onnx_filename, engine_filename=None, max_batch_size=32, max_workspace_size=1 << 30,
                           fp16_mode=True):
    # In node -1 (importModel): INVALID_VALUE: Assertion failed: !_importer_ctx.network()->hasImplicitBatchDimension() && "This version of the ONNX parser only supports TensorRT INetworkDefinitions with an explicit batch dimension. Please ensure the network was created using the EXPLICIT_BATCH NetworkDefinitionCreationFlag."
    # Fix: https://blog.csdn.net/blueblood7/article/details/123697205
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder:
        with builder.create_network(EXPLICIT_BATCH) as network:
            with trt.OnnxParser(network, logger) as parser:
                builder.max_workspace_size = max_workspace_size
                builder.fp16_mode = fp16_mode
                builder.max_batch_size = max_batch_size

                print("Parsing ONNX file.")
                with open(onnx_filename, 'rb') as model:
                    if not parser.parse(model.read()):
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))

                print("Building TensorRT engine. This may take a few minutes.")
                engine = builder.build_cuda_engine(network)

                if engine_filename:
                    with open(engine_filename, 'wb') as f:
                        f.write(engine.serialize())

                return engine, logger


def main(args):
    onnx_path = args.onnx
    engine_path = args.engine
    batch_size = args.batch_size
    use_fp16 = args.fp16

    convert_onnx_to_engine(onnx_path, engine_path, max_batch_size=batch_size, fp16_mode=use_fp16)
    print(f"Save engine to {engine_path}")


if __name__ == '__main__':
    args = parse_opt()
    main(args)
