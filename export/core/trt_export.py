# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 16:46
@File    : trt_exporter.py
@Author  : zj
@Description: 
"""

# core/trt_exporter.py

import subprocess
import os
import onnx


def get_input_profile(onnx_path, batch_size):
    """
    从 ONNX 模型中提取输入信息，生成 trtexec 所需的 shape 参数。

    返回示例:
        input_name = 'input'
        opt_shape = 'input:1x3x224x224'
        max_shape = 'input:8x3x224x224'
    """
    model = onnx.load(onnx_path)
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name

    # 获取输入维度
    dims = input_tensor.type.tensor_type.shape.dim
    dynamic_batch = False

    input_shape = []
    for i, dim in enumerate(dims):
        if dim.HasField("dim_value"):
            val = dim.dim_value
        elif dim.HasField("dim_param"):
            val = 1  # 动态维度使用默认值作为 opt 值
            dynamic_batch = (i == 0)  # 第一维是否是动态 batch？
        else:
            val = 1

        input_shape.append(val)

    # 固定 batch 大小
    input_shape[0] = 1
    max_shape_list = input_shape.copy()
    max_shape_list[0] = batch_size

    def shape_to_str(shape):
        return "x".join(map(str, shape))

    opt_shape_str = f"{input_name}:{shape_to_str(input_shape)}"
    max_shape_str = f"{input_name}:{shape_to_str(max_shape_list)}"

    return input_name, opt_shape_str, max_shape_str, dynamic_batch


def convert_onnx_to_tensorrt(
        onnx_path: str,
        engine_path: str,
        fp16: bool = False,
        batch_size: int = None,  # 动态批量大小，None 表示固定 ONNX 原始值
        workspace: int = 4096,  # 单位 MB，默认 4096 MB = 4GB
        avg_runs: int = 100,  # 性能测试运行次数
        verbose: bool = False
):
    """
    使用 trtexec 将 ONNX 转换为 TensorRT 引擎，支持动态输入形状和性能调优参数。
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--explicitBatch"
    ]

    # 精度设置
    if fp16:
        cmd += ["--fp16", "--inputIOFormats=fp16:chw", "--outputIOFormats=fp16:chw"]

    # 显存限制
    cmd += [f"--workspace={workspace}"]

    if batch_size is not None and batch_size > 1:
        # 输入形状设置
        input_name, opt_shape, max_shape, _ = get_input_profile(onnx_path, batch_size)
        cmd += [
            f"--optShapes={opt_shape}",
            f"--maxShapes={max_shape}",
            f"--minShapes={opt_shape}"
        ]
    else:
        print("ℹ️ 使用 ONNX 模型中定义的输入形状（未指定动态范围）")

    # 性能评估参数
    cmd += [f"--avgRuns={avg_runs}"]

    if verbose:
        print("🚀 Running trtexec command:")
        print(" ".join(cmd))

    print("🔄 Converting ONNX to TensorRT engine...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 无论成功与否，都打印 trtexec 的完整输出日志
    print("📋 TensorRT conversion log:")
    print(result.stdout)

    if result.returncode != 0:
        print("❌ Error during TensorRT conversion:")
        print(result.stdout)
        raise RuntimeError("TensorRT engine conversion failed.")
    else:
        print(f"✅ TensorRT engine saved at: {engine_path}")
