# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : config_generator.py
@Author  : zj
@Description: Triton config.pbtxt auto-generator

Supports ONNX Runtime and TensorRT backends,
auto-detects task type (detect / classify / segment / pose) → auto-sets input/output dims.

Usage:
    >>> from export.triton import TritonConfigGenerator
    >>> gen = TritonConfigGenerator(
    ...     model_name="Detect_COCO_YOLOv8s_TRT",
    ...     backend="tensorrt",
    ...     task="detect",
    ... )
    >>> gen.save("models/triton/")
"""

from typing import Optional, List, Dict, Any


# Task → input/output specifications
_TASK_IO_SPECS: Dict[str, Dict[str, Any]] = {
    "classify": {
        "input": {"name": "image", "data_type": "TYPE_FP32", "dims": [3, 224, 224]},
        "output": [{"name": "output0", "data_type": "TYPE_FP32", "dims": [1000]}],
    },
    "detect": {
        "input": {"name": "image", "data_type": "TYPE_FP32", "dims": [3, 640, 640]},
        "output": [{"name": "output0", "data_type": "TYPE_FP32", "dims": [84, 8400]}],
    },
    "segment": {
        "input": {"name": "image", "data_type": "TYPE_FP32", "dims": [3, 640, 640]},
        "output": [
            {"name": "output0", "data_type": "TYPE_FP32", "dims": [116, 8400]},
            {"name": "output1", "data_type": "TYPE_FP32", "dims": [32, 160, 160]},
        ],
    },
    "pose": {
        "input": {"name": "image", "data_type": "TYPE_FP32", "dims": [3, 640, 640]},
        "output": [{"name": "output0", "data_type": "TYPE_FP32", "dims": [56, 8400]}],
    },
}

_BACKEND_PLATFORMS = {
    "onnxruntime": "onnxruntime_onnx",
    "tensorrt": "tensorrt_plan",
    "onnx": "onnxruntime_onnx",
    "trt": "tensorrt_plan",
}


class TritonConfigGenerator:
    """Triton config.pbtxt generator

    Args:
        model_name: Model name (must match the directory name), e.g. "Detect_COCO_YOLOv8s_TRT"
        backend: Backend type ("onnxruntime" / "tensorrt")
        task: Task type ("classify" / "detect" / "segment" / "pose")
        max_batch_size: Maximum batch size (0 = no batch dimension)
        instance_count: Number of model instances (default 1)
        dynamic_batching: Whether to enable dynamic batching
        preferred_batch_size: Preferred batch sizes for dynamic batching
        max_queue_delay: Maximum queue delay in microseconds
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "tensorrt",
        task: str = "detect",
        max_batch_size: int = 0,
        instance_count: int = 1,
        dynamic_batching: bool = False,
        preferred_batch_size: Optional[List[int]] = None,
        max_queue_delay_microseconds: int = 100,
    ):
        self.model_name = model_name
        self.backend = _BACKEND_PLATFORMS.get(backend, backend)
        self.task = task
        self.max_batch_size = max_batch_size
        self.instance_count = instance_count
        self.dynamic_batching = dynamic_batching
        self.preferred_batch_size = preferred_batch_size or [1, 4, 8]
        self.max_queue_delay = max_queue_delay_microseconds

        if task not in _TASK_IO_SPECS:
            raise ValueError(f"Unknown task {task!r}. Options: {list(_TASK_IO_SPECS.keys())}")

    @property
    def io_spec(self) -> dict:
        return _TASK_IO_SPECS[self.task]

    def generate(self) -> str:
        """Generate config.pbtxt content

        Returns:
            config.pbtxt text
        """
        lines: List[str] = []
        spec = self.io_spec

        lines.append(f'name: "{self.model_name}"')
        lines.append(f'platform: "{self.backend}"')
        lines.append(f"max_batch_size: {self.max_batch_size}")
        lines.append("")

        # Dynamic batching
        if self.dynamic_batching:
            lines.append("dynamic_batching {")
            if self.preferred_batch_size:
                sizes = ", ".join(str(s) for s in self.preferred_batch_size)
                lines.append(f"  preferred_batch_size: [{sizes}]")
            lines.append(f"  max_queue_delay_microseconds: {self.max_queue_delay}")
            lines.append("}")
            lines.append("")

        # Instance group
        if self.instance_count > 1:
            lines.extend([
                "instance_group {",
                f"  count: {self.instance_count}",
                "  kind: KIND_GPU",
                "}",
                "",
            ])

        # Input
        inp = spec["input"]
        lines.append("input [")
        lines.append("  {")
        lines.append(f'    name: "{inp["name"]}"')
        lines.append(f'    data_type: {inp["data_type"]}')
        lines.append(f'    dims: [{", ".join(str(d) for d in inp["dims"])}]')
        lines.append("  }")
        lines.append("]")
        lines.append("")

        # Output
        outputs = spec["output"]
        lines.append("output [")
        for i, out in enumerate(outputs):
            lines.append("  {")
            lines.append(f'    name: "{out["name"]}"')
            lines.append(f'    data_type: {out["data_type"]}')
            lines.append(f'    dims: [{", ".join(str(d) for d in out["dims"])}]')
            lines.append("  }")
            if i < len(outputs) - 1:
                lines.append("  ,")
        lines.append("]")
        lines.append("")

        return "\n".join(lines)

    def save(self, model_repo_dir: str, version: int = 1) -> str:
        """Generate config.pbtxt and save to the model repository directory

        Args:
            model_repo_dir: Model repository root directory
            version: Model version number (directory name)

        Returns:
            config.pbtxt file path
        """
        import os
        from pathlib import Path

        model_dir = Path(model_repo_dir) / self.model_name / str(version)
        model_dir.mkdir(parents=True, exist_ok=True)

        config_path = Path(model_repo_dir) / self.model_name / "config.pbtxt"
        config_path.write_text(self.generate())
        print(f"[Triton] ✅ config.pbtxt saved to {config_path}")

        return str(config_path)


# ==================== CLI ====================

def parse_opt():
    import argparse
    parser = argparse.ArgumentParser(description="Triton config.pbtxt generator")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--backend", type=str, default="tensorrt",
                        choices=["onnxruntime", "tensorrt"], help="Backend type")
    parser.add_argument("--task", type=str, default="detect",
                        choices=["classify", "detect", "segment", "pose"],
                        help="Task type")
    parser.add_argument("--save", type=str, required=True, help="Output directory (model repository root)")
    parser.add_argument("--max-batch", type=int, default=0, help="Maximum batch size")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batching")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of model instances")
    return parser.parse_args()


def main():
    args = parse_opt()
    gen = TritonConfigGenerator(
        model_name=args.model_name,
        backend=args.backend,
        task=args.task,
        max_batch_size=args.max_batch,
        instance_count=args.instance_count,
        dynamic_batching=args.dynamic_batch,
    )
    gen.save(args.save)


if __name__ == "__main__":
    main()
