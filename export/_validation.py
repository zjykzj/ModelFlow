# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : validation.py
@Author  : zj
@Description: ONNX format validation, PT vs ONNX output comparison

All exporters go through this module for unified validation, avoiding duplicate implementations.
"""

from typing import List, Optional, Union, Tuple

import numpy as np
import onnx
import onnxruntime


def check_onnx(onnx_path: str) -> bool:
    """Validate ONNX model format

    Args:
        onnx_path: Path to the ONNX file

    Returns:
        True if validation passes, otherwise raises an exception
    """
    print(f"[Validation] Checking ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        raise ValueError(f"ONNX model validation failed for {onnx_path}: {e}") from e
    print("[Validation] ONNX model check passed.")
    return True


def _to_numpy(tensor):
    """PyTorch Tensor to numpy ndarray (generic utility, avoids forcing import torch)"""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def compare_torch_onnx(
    torch_outputs,
    torch_model,
    onnx_path: str,
    input_tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Compare PyTorch vs ONNX Runtime outputs

    Args:
        torch_outputs: PyTorch model forward output (single Tensor or list/tuple)
        torch_model: PyTorch model (used for checking if torch_outputs needs recalculation)
        onnx_path: Path to the ONNX file
        input_tensor: Input tensor (for ONNX Runtime inference)
        rtol: Relative error tolerance
        atol: Absolute error tolerance

    Returns:
        True if all outputs match, otherwise False
    """
    print("[Validation] Comparing PyTorch vs ONNX Runtime outputs...")

    try:
        ort_session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        print(f"[Validation] ONNX Runtime failed to load: {e}")
        return False

    # Normalize torch_outputs to list
    if not isinstance(torch_outputs, (list, tuple)):
        torch_outputs = [torch_outputs]

    # ONNX inference
    ort_inputs = {ort_session.get_inputs()[0].name: _to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)

    all_ok = True
    for i, (torch_out, ort_out) in enumerate(zip(torch_outputs, ort_outs)):
        try:
            np.testing.assert_allclose(
                _to_numpy(torch_out), ort_out, rtol=rtol, atol=atol
            )
            print(f"[Validation]   Output {i} matches (rtol={rtol}, atol={atol})")
        except AssertionError as e:
            print(f"[Validation]   Output {i} mismatch: {e}")
            all_ok = False

    return all_ok


def compare_output(
    onnx_path: str,
    input_tensor: "torch.Tensor",
    torch_model: "torch.nn.Module",
    torch_outputs: Optional[Union[List, Tuple]] = None,
    **kwargs,
) -> bool:
    """One-stop comparison: auto run forward pass if torch_outputs not provided

    Args:
        onnx_path: Path to the ONNX file
        input_tensor: Input tensor
        torch_model: PyTorch model
        torch_outputs: Optional PyTorch forward output (auto-computed if not passed)
        **kwargs: Passed to compare_torch_onnx

    Returns:
        True if all outputs match, otherwise False
    """
    import torch

    if torch_outputs is None:
        with torch.no_grad():
            torch_outputs = torch_model(input_tensor)

    return compare_torch_onnx(torch_outputs, torch_model, onnx_path, input_tensor, **kwargs)
