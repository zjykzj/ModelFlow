# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : optimize.py
@Author  : zj
@Description: ONNX model optimization utility

Wraps onnx-simplifier to simplify the computation graph:
- Remove redundant Identity nodes
- Merge adjacent Reshape / Transpose
- Eliminate unnecessary Cast operations

Usage:
    >>> from export.onnx import optimize_onnx
    >>> optimize_onnx("model.onnx", "model_simplified.onnx")
"""

from typing import Optional


def optimize_onnx(
    input_path: str,
    output_path: Optional[str] = None,
    skip_optimization: bool = False,
) -> str:
    """Simplify the ONNX model graph structure (wraps onnx-simplifier)

    Prints an install hint if onnx-simplifier is not available.

    Args:
        input_path: Input ONNX file path
        output_path: Output ONNX file path (overwrites input by default)
        skip_optimization: If True, skip simplification and return the input path as-is

    Returns:
        str: Path to the simplified ONNX model
    """
    if skip_optimization:
        return input_path

    if output_path is None:
        output_path = input_path

    try:
        import onnxsim
    except ImportError:
        print("[Optimize] ⚠️  onnx-simplifier not installed. Skipping optimization.")
        print("[Optimize]   Install: pip install onnx-simplifier onnxruntime")
        return input_path

    import onnx

    print(f"[Optimize] Simplifying {input_path} -> {output_path} ...")
    model, _ = onnxsim.simplify(input_path)
    onnx.save(model, output_path)
    print(f"[Optimize] ✅ Simplified model saved to {output_path}")

    return output_path
