# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/10 16:27
@File    : export_onnx_trt_to_triton.py
@Author  : zj
@Description: 
"""

import argparse
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

    :param model_name: Name of the model in Triton
    :param model_type: Type of model ('onnx' or 'tensorrt')
    :param input_shapes: List of input shapes (e.g., [[3, 640, 640]])
    :param output_shapes: List of output shapes
    :param max_batch_size: Max batch size (0 means no batch dimension)
    :param dynamic_batching: Whether to enable dynamic batching
    :param precision: Precision mode ("fp32" or "fp16")
    :param input_name: Input tensor name
    :param output_name: Output tensor name
    :return: Generated config string
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

    :param model_path: Path to the ONNX or TensorRT engine file
    :param model_repo: Root directory of the Triton model repository
    :param model_name: Name of the model in Triton
    :param model_version: Version number (default: 1)
    :param input_shapes: Input dimensions (list of lists)
    :param output_shapes: Output dimensions (list of lists)
    :param max_batch_size: Maximum batch size allowed
    :param dynamic_batching: Enable dynamic batching
    :param precision: Model inference precision
    :param input_name: Name of input tensor
    :param output_name: Name of output tensor
    """

    input_shapes = input_shapes or [[3, 640, 640]]
    output_shapes = output_shapes or [[84, 8400]]

    model_type = "onnx" if model_path.endswith(".onnx") else "tensorrt"

    model_dir = os.path.join(model_repo, model_name)
    version_dir = os.path.join(model_dir, str(model_version))

    # Create necessary directories
    os.makedirs(version_dir, exist_ok=True)

    # Copy model file to correct location
    target_model_file = os.path.join(version_dir, "model.onnx" if model_type == "onnx" else "model.engine")
    shutil.copy(model_path, target_model_file)

    # Generate and write config.pbtxt
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


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Export ONNX or TensorRT models to Triton Inference Server format.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the ONNX or TensorRT (.engine) model file")
    parser.add_argument("--model-repo", type=str, required=True,
                        help="Root path to the Triton model repository")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the model in the Triton server")
    parser.add_argument("--model-version", type=int, default=1,
                        help="Model version number (default: 1)")
    parser.add_argument("--input-shapes", nargs='+', type=int, action='append', default=None,
                        help="Input shapes, e.g., --input-shapes 3 640 640")
    parser.add_argument("--output-shapes", nargs='+', type=int, action='append', default=None,
                        help="Output shapes, e.g., --output-shapes 84 8400")
    parser.add_argument("--max-batch-size", type=int, default=0,
                        help="Max batch size (0 means no batch dimension)")
    parser.add_argument("--no-dynamic-batching", action="store_false", dest="dynamic_batching",
                        help="Disable dynamic batching")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp32",
                        help="Inference precision (default: fp32)")
    parser.add_argument("--input-name", type=str, default="input",
                        help="Name of input tensor (default: input)")
    parser.add_argument("--output-name", type=str, default="output",
                        help="Name of output tensor (default: output)")

    return parser.parse_args()


def main():
    args = parse_args()

    input_shapes = args.input_shapes if args.input_shapes else [[3, 640, 640]]
    output_shapes = args.output_shapes if args.output_shapes else [[84, 8400]]

    export_to_triton(
        model_path=args.model_path,
        model_repo=args.model_repo,
        model_name=args.model_name,
        model_version=args.model_version,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        max_batch_size=args.max_batch_size,
        dynamic_batching=args.dynamic_batching,
        precision=args.precision,
        input_name=args.input_name,
        output_name=args.output_name
    )


if __name__ == "__main__":
    main()
