# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : build_int8_pycuda.py
@Author  : zj
@Description: TensorRT INT8 引擎构建器（PyCUDA 校准器）

适用场景：NVIDIA Jetson、嵌入式设备、Docker 精简镜像。
核心优势：极致轻量，无需 PyTorch，直接绑定 CUDA 驱动。

用法：
    >>> from export.tensorrt import build_int8_engine_pycuda
    >>> build_int8_engine_pycuda(
    ...     onnx_path="model.onnx",
    ...     calib_dir="calib_data",
    ...     output_path="model_int8.engine",
    ... )

CLI：
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
    """使用 PyCUDA 校准器构建 INT8 引擎

    在调用此函数前，调用者需确保 pycuda.autoinit 已初始化。

    Args:
        onnx_path: ONNX 模型路径
        calib_dir: 校准数据目录（包含 .bin 文件）
        output_path: 引擎保存路径
        input_shape: 输入 shape (N, C, H, W)
        workspace: 工作空间大小 (GB)
        cache_file: 校准缓存文件名

    Returns:
        True 构建成功
    """
    if not os.path.exists(onnx_path):
        print(f"[TRT INT8] ❌ ONNX not found: {onnx_path}")
        return False

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

    # 解析 ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False
    print("[TRT INT8] ✅ ONNX parsed")

    # 配置 INT8
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

    # 构建
    print(f"[TRT INT8] Building engine (this may take a while)...")
    try:
        serialized = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"[TRT INT8] ❌ Build failed: {e}")
        return False

    if serialized is None:
        print("[TRT INT8] ❌ Build returned None")
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[TRT INT8] ✅ Engine saved to {output_path} ({size_mb:.2f} MB)")
    return True


# ==================== CLI ====================


def parse_opt():
    parser = argparse.ArgumentParser(description="INT8 引擎构建 (PyCUDA 校准器)")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--calib_dir", type=str, required=True, help="校准数据目录 (.bin)")
    parser.add_argument("--output", type=str, default="model_int8.engine", help="引擎保存路径")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 640, 640],
                        help="N C H W")
    parser.add_argument("--workspace", type=int, default=4, help="工作空间 (GB)")
    parser.add_argument("--cache_file", type=str, default="calib_pycuda.cache",
                        help="校准缓存文件名")
    return parser.parse_args()


def main():
    # PyCUDA 初始化（在 CLI 模式下）
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
