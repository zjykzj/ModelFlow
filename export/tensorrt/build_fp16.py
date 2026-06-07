# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : build_fp16.py
@Author  : zj
@Description: TensorRT FP16 引擎构建器

提供 trtexec 封装和 TensorRT Python API 两种构建方式。

用法：
    >>> from export2.tensorrt import build_fp16_engine
    >>> build_fp16_engine("model.onnx", "model_fp16.engine")
    >>> build_fp16_engine("model.onnx", "model_fp16.engine",
    ...                   use_trtexec=False, workspace=4)

CLI:
    python3 -m export2.tensorrt.build_fp16 --onnx model.onnx --save model_fp16.engine
"""

import os
import sys
import argparse
import subprocess
from typing import Optional, Tuple


def _build_via_trtexec(
    onnx_path: str,
    output_path: str,
    workspace: int = 4,
    shapes: Optional[str] = None,
) -> bool:
    """通过 trtexec 工具构建 FP16 引擎

    Args:
        onnx_path: ONNX 模型路径
        output_path: 引擎保存路径
        workspace: 工作空间大小（GB）
        shapes: 动态 shape 配置字符串，如 "input:1x3x640x640"

    Returns:
        True 构建成功
    """
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace * 1024}",
        "--fp16",
    ]
    if shapes:
        cmd.append(f"--minShapes={shapes}")
        cmd.append(f"--optShapes={shapes}")
        cmd.append(f"--maxShapes={shapes}")

    print(f"[TRT FP16] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"[TRT FP16] ❌ trtexec failed with code {result.returncode}")
        return False
    return True


def _build_via_python_api(
    onnx_path: str,
    output_path: str,
    workspace: int = 4,
    shapes: Optional[Tuple[int, int, int, int]] = None,
) -> bool:
    """通过 TensorRT Python API 构建 FP16 引擎

    Args:
        onnx_path: ONNX 模型路径
        output_path: 引擎保存路径
        workspace: 工作空间大小（GB）
        shapes: 输入 shape (N, C, H, W)

    Returns:
        True 构建成功
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))

    print(f"[TRT FP16] Building engine via Python API (workspace={workspace}GB)...")
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        print("[TRT FP16] ❌ Build failed")
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[TRT FP16] ✅ Engine saved to {output_path} ({size_mb:.2f} MB)")
    return True


def build_fp16_engine(
    onnx_path: str,
    output_path: str,
    workspace: int = 4,
    use_trtexec: bool = True,
    shapes: Optional[str] = None,
) -> bool:
    """构建 TensorRT FP16 引擎

    默认优先尝试 trtexec（简单可靠），失败则回退到 Python API。
    如果 trtexec 完全不可用，设置 use_trtexec=False。

    Args:
        onnx_path: ONNX 模型路径
        output_path: 引擎保存路径
        workspace: 工作空间大小（GB）
        use_trtexec: 是否优先使用 trtexec
        shapes: 动态 shape（仅 trtexec 模式，"input:1x3x640x640"）

    Returns:
        True 构建成功
    """
    if not os.path.exists(onnx_path):
        print(f"[TRT FP16] ❌ ONNX not found: {onnx_path}")
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output_path = os.path.abspath(output_path)

    print("=" * 60)
    print(f"[TRT FP16] Building FP16 engine")
    print(f"  ONNX:  {onnx_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    if use_trtexec:
        if _build_via_trtexec(onnx_path, output_path, workspace, shapes):
            return True
        print("[TRT FP16] trtexec failed, falling back to Python API...")

    return _build_via_python_api(onnx_path, output_path, workspace)


# ==================== CLI ====================


def parse_opt():
    parser = argparse.ArgumentParser(description="TensorRT FP16 引擎构建")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--save", type=str, required=True, help="引擎保存路径")
    parser.add_argument("--workspace", type=int, default=4, help="工作空间 (GB)")
    parser.add_argument(
        "--no-trtexec", action="store_true",
        help="不使用 trtexec，直接使用 Python API",
    )
    parser.add_argument("--shapes", type=str, default=None, help="动态 shape (input:1x3x640x640)")
    return parser.parse_args()


def main():
    args = parse_opt()
    success = build_fp16_engine(
        onnx_path=args.onnx,
        output_path=args.save,
        workspace=args.workspace,
        use_trtexec=not args.no_trtexec,
        shapes=args.shapes,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
