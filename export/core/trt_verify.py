# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 16:46
@File    : trt_verify.py
@Author  : zj
@Description: 
"""

import numpy as np

from models.onnx_classifier import ONNXClassifier
from models.trt_classifier import TRTClassifier


def verify_trt_output(onnx_path: str, engine_path: str, batch_size: int = 1, is_fp16: bool = False):
    """
    Verify that the TensorRT engine can run and produce output of correct shape.

    Uses ONNXRuntime and TensorRT classifiers for inference comparison.
    """
    # 强制 ONNX 使用 float32，TRT 可选 float16
    dummy_input = load_dummy_input_from_onnx(onnx_path, batch_size=batch_size, dtype=np.float32)

    # Step 2: ONNX Inference
    onnx_model = ONNXClassifier(onnx_path)
    onnx_output = onnx_model.predict(dummy_input)

    # Step 3: TensorRT Inference
    trt_model = TRTClassifier(engine_path)
    trt_input = dummy_input.astype(np.float16 if is_fp16 else np.float32)
    trt_output = trt_model.predict(trt_input)

    # Step 4: Compare shapes and values
    assert onnx_output.shape == trt_output.shape, \
        f"Shape mismatch: ONNX={onnx_output.shape}, TRT={trt_output.shape}"

    print("✅ TensorRT inference completed.")
    print("Output shape:", trt_output.shape)
    print("First 5 ONNX values:", onnx_output[:5])
    print("First 5 TRT values :", trt_output[:5])

    diff = np.abs(onnx_output - trt_output).max()
    print(f"Maximum absolute difference between ONNX and TRT outputs: {diff:.6f}")
    if diff < 1e-3:
        print("🎉 Difference is within acceptable range.")
    else:
        print("⚠️ Warning: Large difference detected. Consider checking FP16 precision or input shape consistency.")


def load_dummy_input_from_onnx(onnx_path: str, batch_size: int = 1, dtype=np.float32) -> np.ndarray:
    import onnx
    model = onnx.load(onnx_path)
    input_shape = [dim.dim_value if dim.HasField('dim_value') else -1 for dim in
                   model.graph.input[0].type.tensor_type.shape.dim]
    assert input_shape[0] == -1 or input_shape[0] == batch_size, "Input shape mismatch with batch size."

    input_shape[0] = batch_size
    return np.random.rand(*input_shape).astype(dtype)
