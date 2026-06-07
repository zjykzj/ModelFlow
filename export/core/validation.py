# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : validation.py
@Author  : zj
@Description: ONNX 格式校验、PT vs ONNX 输出对比

所有导出器通过此模块统一验证流程，避免重复实现。
"""

from typing import List, Optional, Union, Tuple

import numpy as np
import onnx
import onnxruntime


def check_onnx(onnx_path: str) -> bool:
    """检查 ONNX 模型格式是否合法

    Args:
        onnx_path: ONNX 文件路径

    Returns:
        True 验证通过，否则抛出异常
    """
    print(f"[Validation] Checking ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("[Validation] ✅ ONNX model check passed.")
    return True


def _to_numpy(tensor):
    """PyTorch Tensor → numpy ndarray（通用工具，避免强制 import torch）"""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def compare_torch_onnx(
    torch_outputs,
    torch_model,
    onnx_path: str,
    input_tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """对比 PyTorch 与 ONNX Runtime 的输出

    Args:
        torch_outputs: PyTorch 模型前向输出（单 Tensor 或 list/tuple）
        torch_model: PyTorch 模型（用于检查是否需要对 torch_outputs 重新计算）
        onnx_path: ONNX 文件路径
        input_tensor: 输入张量（用于 ONNX Runtime 推理）
        rtol: 相对误差容限
        atol: 绝对误差容限

    Returns:
        True 全部匹配，否则 False
    """
    print("[Validation] Comparing PyTorch vs ONNX Runtime outputs...")

    try:
        ort_session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        print(f"[Validation] ❌ ONNX Runtime failed to load: {e}")
        return False

    # 统一 torch_outputs 为 list
    if not isinstance(torch_outputs, (list, tuple)):
        torch_outputs = [torch_outputs]

    # ONNX 推理
    ort_inputs = {ort_session.get_inputs()[0].name: _to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)

    all_ok = True
    for i, (torch_out, ort_out) in enumerate(zip(torch_outputs, ort_outs)):
        try:
            np.testing.assert_allclose(
                _to_numpy(torch_out), ort_out, rtol=rtol, atol=atol
            )
            print(f"[Validation]   ✅ Output {i} matches (rtol={rtol}, atol={atol})")
        except AssertionError as e:
            print(f"[Validation]   ❌ Output {i} mismatch: {e}")
            all_ok = False

    return all_ok


def compare_output(
    onnx_path: str,
    input_tensor: "torch.Tensor",
    torch_model: "torch.nn.Module",
    torch_outputs: Optional[Union[List, Tuple]] = None,
    **kwargs,
) -> bool:
    """一站式对比：如果未提供 torch_outputs，自动运行一次前向传播

    Args:
        onnx_path: ONNX 文件路径
        input_tensor: 输入张量
        torch_model: PyTorch 模型
        torch_outputs: 可选的 PyTorch 前向输出（不传则自动计算）
        **kwargs: 传给 compare_torch_onnx

    Returns:
        True 全部匹配，否则 False
    """
    import torch

    if torch_outputs is None:
        with torch.no_grad():
            torch_outputs = torch_model(input_tensor)

    return compare_torch_onnx(torch_outputs, torch_model, onnx_path, input_tensor, **kwargs)
