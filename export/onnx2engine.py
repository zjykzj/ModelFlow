# -*- coding: utf-8 -*-

"""
@Time    : 2023/10/29 15:55
@File    : onnx2engine.py
@Author  : zj
@Description: Convert onnx model to tensorrt format
See:
1. https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/onnx_helper.py

Usage: Convert ONNX Resnet50 to Engine:
    $ python onnx2engine.py resnet50_pytorch.onnx resnet50_pytorch.engine

Usage: Convert to FP16 precision:
    $ python onnx2engine.py resnet50_pytorch.onnx resnet50_pytorch_fp16.engine --fp16

[TensorRT] ERROR: Network has dynamic or shape inputs, but no optimization profile has been defined.
Fix: Using the trtexec command-line tool for dynamic onnx model conversion
"""

import argparse
import onnxruntime

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

print("TensorRT version: {}".format(trt.__version__))


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
    # ERROR: In node -1 (importModel): INVALID_VALUE: Assertion failed: !_importer_ctx.network()->hasImplicitBatchDimension() && "This version of the ONNX parser only supports TensorRT INetworkDefinitions with an explicit batch dimension. Please ensure the network was created using the EXPLICIT_BATCH NetworkDefinitionCreationFlag."
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


class ONNXClassifierWrapper():
    def __init__(self, file, num_classes, target_dtype=np.float32):
        self.target_dtype = target_dtype
        self.num_classes = num_classes
        self.load(file)

        self.stream = None

    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes,
                               dtype=self.target_dtype)  # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def predict(self, batch):  # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return self.output


def get_onnx_output(onnx_path):
    ort_session = onnxruntime.InferenceSession(onnx_path,
                                               providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                          'AzureExecutionProvider', 'CPUExecutionProvider'])
    print("Onnx info:")
    print(f"    input: {ort_session.get_inputs()[0]}")
    print(f"    output: {ort_session.get_outputs()[0]}")

    assert isinstance(ort_session.get_inputs()[0].shape[0], int)
    np.random.seed(10)
    x = np.random.randn(*ort_session.get_inputs()[0].shape).astype(np.float32)
    num_classes = ort_session.get_outputs()[0].shape[1]

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)
    print(x.shape, ort_outs[0].shape)

    return x, num_classes, ort_outs[0]


def check_engine(x, num_classes, ort_out, engine_path, is_fp16=False):
    target_type = np.float16 if is_fp16 else np.float32
    engine_model = ONNXClassifierWrapper(engine_path, num_classes, target_dtype=target_type)

    trt_out = engine_model.predict(x.astype(target_type)).reshape(ort_out.shape)

    # compare ONNX Runtime and TensorRT results
    np.testing.assert_allclose(ort_out, trt_out, rtol=1e-02, atol=1e-02)
    print("Exported model has been tested with ONNXRuntime and TensorRT, and the result looks good!")


def main(args):
    onnx_path = args.onnx
    engine_path = args.engine
    batch_size = args.batch_size
    use_fp16 = args.fp16

    print("=> convert_onnx_to_engine")
    convert_onnx_to_engine(onnx_path, engine_path, max_batch_size=batch_size, fp16_mode=use_fp16)
    print(f"Save engine to {engine_path}")

    if not use_fp16:
        print("=> get_onnx_output")
        x, num_classes, ort_out = get_onnx_output(onnx_path)
        print("=> check_engine")
        check_engine(x, num_classes, ort_out, engine_path, is_fp16=use_fp16)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
