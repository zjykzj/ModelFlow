# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : build_int8_pycuda.py
@Author  : zj
@Description: TensorRT INT8 Engine Builder (PyCUDA Calibrator)

Use cases: NVIDIA Jetson, embedded devices, slim Docker images.
Key advantage: Ultra-lightweight, no PyTorch dependency, binds CUDA driver directly.

Usage:
    >>> from export.tensorrt import build_int8_engine_pycuda
    >>> build_int8_engine_pycuda(
    ...     onnx_path="model.onnx",
    ...     calib_dir="calib_data",
    ...     output_path="model_int8.engine",
    ... )

CLI:
    python3 -m export.tensorrt.build_int8_pycuda \\
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

from .calibrator import PyCudaCalibrator


def build_int8_engine_pycuda(
    onnx_path: str,
    calib_dir: str,
    output_path: str,
    input_shape=(1, 3, 640, 640),
    workspace: int = 4,
    cache_file: str = "calib_pycuda.cache",
) -> bool:
    """Build an INT8 engine using the PyCUDA calibrator.

    The caller must ensure pycuda.autoinit has been initialized before
    invoking this function.

    Args:
        onnx_path: Path to the ONNX model.
        calib_dir: Calibration data directory (containing .bin files).
        output_path: Path to save the built engine.
        input_shape: Input shape (N, C, H, W).
        workspace: Workspace size in GB.
        cache_file: Calibration cache file name.

    Returns:
        True if the build succeeded.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print("=" * 60)
    print(f"[TRT INT8] Building INT8 engine (PyCUDA calibrator)")
    print(f"  ONNX:    {onnx_path}")
    print(f"  Shape:   {input_shape}")
    print(f"  Calib:   {calib_dir}")
    print(f"  Cache:   {cache_file}")
    print("=" * 60)

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
    print("[TRT INT8] ✅ ONNX parsed")

    # Configure INT8
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)

    calibrator = PyCudaCalibrator(
        calib_data_dir=calib_dir,
        input_shape=tuple(input_shape),
        cache_file=cache_file,
    )
    config.int8_calibrator = calibrator

    # Build
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
    parser = argparse.ArgumentParser(description="INT8 Engine Build (PyCUDA Calibrator)")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--calib_dir", type=str, required=True, help="Calibration data directory (.bin)")
    parser.add_argument("--output", type=str, default="model_int8.engine", help="Path to save the engine")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 640, 640],
                        help="N C H W")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size (GB)")
    parser.add_argument("--cache_file", type=str, default="calib_pycuda.cache",
                        help="Calibration cache file name")
    return parser.parse_args()


def main():
    # Initialize PyCUDA (CLI mode)
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        device = cuda.Device(0)
        print(f"[TRT INT8] 🖥  Device: {device.name()}")
    except ImportError:
        print("[TRT INT8] ❌ pycuda not installed")
        print("  Install: pip install pycuda")
        print("  Or use build_int8_engine_torch (with PyTorch)")
        sys.exit(1)

    args = parse_opt()
    success = build_int8_engine_pycuda(
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
