# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : optimize.py
@Author  : zj
@Description: ONNX 模型优化工具

封装 onnx-simplifier 的调用，简化计算图：
- 移除冗余 Identity 节点
- 合并相邻的 Reshape / Transpose
- 消除无用 Cast 操作

用法：
    >>> from export2.onnx import optimize_onnx
    >>> optimize_onnx("model.onnx", "model_simplified.onnx")
"""

from typing import Optional


def optimize_onnx(
    input_path: str,
    output_path: Optional[str] = None,
    skip_optimization: bool = False,
) -> str:
    """简化 ONNX 模型图结构（封装 onnx-simplifier）

    如果 onnx-simplifier 未安装，会给出安装提示。

    Args:
        input_path: 输入 ONNX 路径
        output_path: 输出 ONNX 路径（默认覆盖原文件）
        skip_optimization: 如为 True，则不做简化直接返回原路径

    Returns:
        str: 简化后的 ONNX 路径
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
