# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/10 16:36
@File    : triton_exporter.py
@Author  : zj
@Description: 
"""

import os
import shutil


def generate_config_pbtxt(
        model_name: str,
        model_type: str,  # 'onnx' or 'tensorrt'
        input_shapes: list,
        output_shapes: list,
        max_batch_size: int = 0,
        dynamic_batching: bool = True,
        precision: str = "fp32",
        input_name: str = "input",
        output_name: str = "output"
):
    """
    Generate config.pbtxt content for Triton Inference Server.
    """
    config = f"""name: "{model_name}"
platform: "{'onnxruntime_onnx' if model_type == 'onnx' else 'tensorrt_plan'}"
max_batch_size: {max_batch_size}
"""

    if dynamic_batching:
        config += """dynamic_batching {
  preferred_batch_size: [1, 2, 4]
  max_queue_delay_microseconds: 100
}
"""

    config += "input [\n"
    for idx, shape in enumerate(input_shapes):
        config += f""" {{
  name: "{input_name}_{idx}" if len(input_shapes) > 1 else "{input_name}"
  data_type: {"TYPE_FP32" if precision == "fp32" else "TYPE_FP16"}
  dims: {list(shape)}
}}"""
    config += "\n]\n"

    config += "output [\n"
    for idx, shape in enumerate(output_shapes):
        config += f""" {{
  name: "{output_name}_{idx}" if len(output_shapes) > 1 else "{output_name}"
  data_type: {"TYPE_FP32" if precision == "fp32" else "TYPE_FP16"}
  dims: {list(shape)}
}}"""
    config += "\n]\n"

    return config


def export_to_triton(
        model_path: str,
        model_repo: str,
        model_name: str,
        model_version: int = 1,
        input_shapes: list = None,
        output_shapes: list = None,
        max_batch_size: int = 0,
        dynamic_batching: bool = True,
        precision: str = "fp32",
        input_name: str = "input",
        output_name: str = "output"
):
    """
    Export ONNX or TensorRT model to Triton Model Repository format.
    """

    input_shapes = input_shapes or [[3, 224, 224]]
    output_shapes = output_shapes or [[1, 1000]]

    model_type = "onnx" if model_path.endswith(".onnx") else "tensorrt"

    model_dir = os.path.join(model_repo, model_name)
    version_dir = os.path.join(model_dir, str(model_version))

    os.makedirs(version_dir, exist_ok=True)

    target_model_file = os.path.join(version_dir, "model.onnx" if model_type == "onnx" else "model.engine")
    shutil.copy(model_path, target_model_file)

    config_str = generate_config_pbtxt(
        model_name=model_name,
        model_type=model_type,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        max_batch_size=max_batch_size,
        dynamic_batching=dynamic_batching,
        precision=precision,
        input_name=input_name,
        output_name=output_name
    )

    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_str)

    print(f"Triton model exported to: {model_dir}")
    print("✅ Done")
