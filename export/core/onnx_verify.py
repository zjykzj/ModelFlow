# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 15:09
@File    : verify.py
@Author  : zj
@Description: 
"""

import torch
import numpy as np

import onnx
import onnxruntime as ort


def to_numpy(tensor):
    """
    Convert a PyTorch tensor to NumPy array.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    return tensor.detach().cpu().numpy()


def print_onnx_model_info(onnx_path: str):
    """
    Print basic information about the ONNX model.

    Args:
        onnx_path (str): Path to the ONNX model file.
    """
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    print("🧾 ONNX Model Summary:")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Producer name: {model.producer_name}")
    print(f"  Producer version: {model.producer_version}")

    # Inputs
    print("\n📥 Inputs:")
    for i, input in enumerate(model.graph.input):
        shape = [dim.dim_value if dim.HasField('dim_value') else -1 for dim in input.type.tensor_type.shape.dim]
        dtype = input.type.tensor_type.elem_type
        print(f"  Input {i}: {input.name} | Shape: {shape} | Data Type: {dtype}")

    # Outputs
    print("\n📤 Outputs:")
    for i, output in enumerate(model.graph.output):
        shape = [dim.dim_value if dim.HasField('dim_value') else -1 for dim in output.type.tensor_type.shape.dim]
        dtype = output.type.tensor_type.elem_type
        print(f"  Output {i}: {output.name} | Shape: {shape} | Data Type: {dtype}")


def validate_onnx_model(onnx_path: str):
    """
    Validate the ONNX model structure.

    Args:
        onnx_path (str): Path to the ONNX model file.
    """
    onnx.checker.check_model(onnx_path)
    print("🧾 ONNX model structure validated.")


def verify_torch_onnx(torch_model, input_tensor, onnx_path):
    """
    Verify numerical consistency between PyTorch and ONNX models.

    Args:
        torch_model (nn.Module): Original PyTorch model.
        input_tensor (torch.Tensor): Dummy input used for inference.
        onnx_path (str): Path to the exported ONNX model.
    """
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(input_tensor).detach().cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    inputs = {session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_out = session.run(None, inputs)[0]

    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-5)
    print("✅ PyTorch output matches ONNX output.")
