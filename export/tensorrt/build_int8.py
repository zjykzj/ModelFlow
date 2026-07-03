# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : build_int8.py
@Author  : zj
@Description: TensorRT INT8 engine builder (PyTorch calibrator)

Suitable for RTX servers, AutoDL cloud hosts, and environments with PyTorch already installed.
Core advantage: no pycuda required, leverages the existing PyTorch environment.

Usage:
    >>> from export.tensorrt import build_int8_engine_torch
    >>> build_int8_engine_torch(
    ...     onnx_path="model.onnx",
    ...     calib_dir="calib_data",
    ...     output_path="model_int8.engine",
    ...     input_shape=(1, 3, 640, 640),
    ... )

CLI:
    python3 -m export.tensorrt.build_int8 \\
        --onnx model.onnx \\
        --calib_dir ./calib_data \\
        --output model_int8.engine \\
        --input_shape 1 3 640 640
"""

import os
import sys
import argparse

try:
    import tensorrt as trt
except ImportError as e:
    raise ImportError(
        "TensorRT is required for INT8 engine building. Install: pip install tensorrt\n"
        "For FP16 engines without TensorRT Python API, use: python3 -m export.tensorrt.build_fp16"
    ) from e

from .calibrator import TorchCalibrator


def build_int8_engine_torch(
    onnx_path: str,
    calib_dir: str,
    output_path: str,
    input_shape=(1, 3, 640, 640),
    workspace: int = 4,
    cache_file: str = "calib_torch.cache",
) -> bool:
    """Build an INT8 engine using the PyTorch calibrator.

    Args:
        onnx_path: Path to the ONNX model.
        calib_dir: Directory containing calibration data (.bin files).
        output_path: Path for saving the engine.
        input_shape: Input shape (N, C, H, W).
        workspace: Workspace size in GB.
        cache_file: Calibration cache file name.

    Returns:
        True on successful build.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for INT8 engine build")

    device = torch.cuda.get_device_name(0)
    print("=" * 60)
    print(f"[TRT INT8] Building INT8 engine (PyTorch calibrator)")
    print(f"  Device:  {device}")
    print(f"  ONNX:    {onnx_path}")
    print(f"  Shape:   {input_shape}")
    print(f"  Calib:   {calib_dir}")
    print("=" * 60)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"ONNX parsing failed for {onnx_path}: {errors}"
            )
    print("[TRT INT8] ✅ ONNX parsed")

    # Configure INT8
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)

    calibrator = TorchCalibrator(
        calib_data_dir=calib_dir,
        input_shape=tuple(input_shape),
        cache_file=cache_file,
    )
    config.int8_calibrator = calibrator

    # Build engine
    print(f"[TRT INT8] Building engine (this may take a while)...")
    try:
        serialized = builder.build_serialized_network(network, config)
    except Exception as e:
        raise RuntimeError(
            f"TensorRT INT8 build failed for {onnx_path}: {e}"
        ) from e

    if serialized is None:
        raise RuntimeError(
            f"TensorRT INT8 build returned None for {onnx_path} "
            f"(likely insufficient calibration data)"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[TRT INT8] ✅ Engine saved to {output_path} ({size_mb:.2f} MB)")
    return True


# ==================== CLI ====================


def parse_opt():
    parser = argparse.ArgumentParser(description="INT8 engine builder (PyTorch calibrator)")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--calib_dir", type=str, required=True, help="Calibration data directory (.bin)")
    parser.add_argument("--output", type=str, default="model_int8.engine", help="Output engine path")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 640, 640],
                        help="N C H W")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size in GB")
    parser.add_argument("--cache_file", type=str, default="calib_torch.cache",
                        help="Calibration cache file name")
    return parser.parse_args()


def main():
    args = parse_opt()
    success = build_int8_engine_torch(
        onnx_path=args.onnx,
        calib_dir=args.calib_dir,
        output_path=args.output,
        input_shape=tuple(args.input_shape),
        workspace=args.workspace,
        cache_file=args.cache_file,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
