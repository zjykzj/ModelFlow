# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : config_generator.py
@Author  : zj
@Description: Triton config.pbtxt 自动生成器

支持 ONNX Runtime 和 TensorRT 两种后端，
任务自动感知（detect / classify / segment / pose）→ 自动设置输入输出 dims。

用法：
    >>> from export2.triton import TritonConfigGenerator
    >>> gen = TritonConfigGenerator(
    ...     model_name="Detect_COCO_YOLOv8s_TRT",
    ...     backend="tensorrt",
    ...     task="detect",
    ... )
    >>> gen.save("models/triton/")
"""

from typing import Optional, List, Dict, Any


# 任务 → 输入输出规格
_TASK_IO_SPECS: Dict[str, Dict[str, Any]] = {
    "classify": {
        "input": {"name": "image", "data_type": "TYPE_FP32", "dims": [3, 224, 224]},
        "output": {"name": "output0", "data_type": "TYPE_FP32", "dims": [1000]},
    },
    "detect": {
        "input": {"name": "image", "data_type": "TYPE_FP32", "dims": [3, 640, 640]},
        "output": {"name": "output0", "data_type": "TYPE_FP32", "dims": [84, 8400]},
    },
    "segment": {
        "input": {"name": "image", "data_type": "TYPE_FP32", "dims": [3, 640, 640]},
        "output": {"name": "output0", "data_type": "TYPE_FP32", "dims": [116, 8400]},
    },
    "pose": {
        "input": {"name": "image", "data_type": "TYPE_FP32", "dims": [3, 640, 640]},
        "output": {"name": "output0", "data_type": "TYPE_FP32", "dims": [56, 8400]},
    },
}

_BACKEND_PLATFORMS = {
    "onnxruntime": "onnxruntime_onnx",
    "tensorrt": "tensorrt_plan",
    "onnx": "onnxruntime_onnx",
    "trt": "tensorrt_plan",
}


class TritonConfigGenerator:
    """Triton config.pbtxt 生成器

    Args:
        model_name: 模型名称（须与目录名一致），如 "Detect_COCO_YOLOv8s_TRT"
        backend: 后端类型 ("onnxruntime" / "tensorrt")
        task: 任务类型 ("classify" / "detect" / "segment" / "pose")
        max_batch_size: 最大 batch（0=无 batch 维度）
        instance_count: 模型实例数（默认 1）
        dynamic_batching: 是否启用 dynamic batching
        preferred_batch_size: 动态 batch 偏好
        max_queue_delay: 最大队列延迟（微秒）
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
        """生成 config.pbtxt 内容

        Returns:
            config.pbtxt 文本
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
        out = spec["output"]
        lines.append("output [")
        lines.append("  {")
        lines.append(f'    name: "{out["name"]}"')
        lines.append(f'    data_type: {out["data_type"]}')
        lines.append(f'    dims: [{", ".join(str(d) for d in out["dims"])}]')
        lines.append("  }")
        lines.append("]")
        lines.append("")

        return "\n".join(lines)

    def save(self, model_repo_dir: str, version: int = 1) -> str:
        """生成 config.pbtxt 并保存到模型仓库目录

        Args:
            model_repo_dir: 模型仓库根目录
            version: 模型版本号（目录名）

        Returns:
            config.pbtxt 文件路径
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
    parser = argparse.ArgumentParser(description="Triton config.pbtxt 生成")
    parser.add_argument("--model-name", type=str, required=True, help="模型名称")
    parser.add_argument("--backend", type=str, default="tensorrt",
                        choices=["onnxruntime", "tensorrt"], help="后端类型")
    parser.add_argument("--task", type=str, default="detect",
                        choices=["classify", "detect", "segment", "pose"],
                        help="任务类型")
    parser.add_argument("--save", type=str, required=True, help="输出目录（模型仓库根目录）")
    parser.add_argument("--max-batch", type=int, default=0, help="最大 batch")
    parser.add_argument("--dynamic-batch", action="store_true", help="启用 dynamic batching")
    parser.add_argument("--instance-count", type=int, default=1, help="模型实例数")
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
