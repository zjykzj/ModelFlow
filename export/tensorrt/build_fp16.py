# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : build_fp16.py
@Author  : zj
@Description: TensorRT FP16 engine builder

Provides trtexec wrapper and TensorRT Python API construction methods.

Usage:
    >>> from export.tensorrt import build_fp16_engine
    >>> build_fp16_engine("model.onnx", "model_fp16.engine")
    >>> build_fp16_engine("model.onnx", "model_fp16.engine",
    ...                   use_trtexec=False, workspace=4)

CLI:
    python3 -m export.tensorrt.build_fp16 --onnx model.onnx --save model_fp16.engine
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
    """Build FP16 engine via trtexec tool

    Args:
        onnx_path: Path to the ONNX model
        output_path: Engine save path
        workspace: Workspace size (GB)
        shapes: Dynamic shape config string, e.g. "input:1x3x640x640"

    Returns:
        True if build succeeded
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
        raise RuntimeError(
            f"trtexec failed with code {result.returncode} for {onnx_path}"
        )
    return True


def _build_via_python_api(
    onnx_path: str,
    output_path: str,
    workspace: int = 4,
    shapes: Optional[Tuple[int, int, int, int]] = None,
) -> bool:
    """Build FP16 engine via TensorRT Python API

    Args:
        onnx_path: Path to the ONNX model
        output_path: Engine save path
        workspace: Workspace size (GB)
        shapes: Input shape (N, C, H, W)

    Returns:
        True if build succeeded
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"ONNX parsing failed for {onnx_path}: {errors}"
            )

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))

    print(f"[TRT FP16] Building engine via Python API (workspace={workspace}GB)...")
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError(f"TensorRT FP16 build failed for {onnx_path}")

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
    """Build TensorRT FP16 engine

    Defaults to trying trtexec first (simple and reliable), falling back
    to the Python API if it fails. Set use_trtexec=False if trtexec is
    completely unavailable.

    Args:
        onnx_path: Path to the ONNX model
        output_path: Engine save path
        workspace: Workspace size (GB)
        use_trtexec: Whether to prefer trtexec
        shapes: Dynamic shape string (trtexec mode only, "input:1x3x640x640")

    Returns:
        True if build succeeded
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

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

    success = _build_via_python_api(onnx_path, output_path, workspace)
    if not success:
        raise RuntimeError(f"TensorRT FP16 build failed for {onnx_path}")
    return True


# ==================== CLI ====================


def parse_opt():
    parser = argparse.ArgumentParser(description="TensorRT FP16 engine builder")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--save", type=str, required=True, help="Engine save path")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size (GB)")
    parser.add_argument(
        "--no-trtexec", action="store_true",
        help="Skip trtexec, use Python API directly",
    )
    parser.add_argument("--shapes", type=str, default=None, help="Dynamic shape string (input:1x3x640x640)")
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
