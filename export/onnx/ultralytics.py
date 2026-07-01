# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : ultralytics.py
@Author  : zj
@Description: Ultralytics YOLO model ONNX exporter

Supports YOLOv8 / YOLO11 / future versions, covering detect / segment / classify / pose tasks.
Underlying implementation uses Ultralytics' YOLO().export(); this module provides CLI wrapper and opset strategy.

Typical usage:
    >>> from export.onnx import UltralyticsExporter
    >>> exporter = UltralyticsExporter("yolov8s")
    >>> onnx_path = exporter.export_onnx("yolov8s.onnx")

CLI:
    python3 -m export.onnx.ultralytics yolov8s
    python3 -m export.onnx.ultralytics yolov8s-seg --opset 12
"""

import argparse
from typing import Optional

from export._base import BaseExporter
from export._validation import check_onnx, compare_output


def get_latest_opset() -> int:
    """Return the second-to-latest ONNX opset supported by the current PyTorch

    Uses max - 1 rather than the latest version for stability (the latest opset may have unresolved bugs).
    """
    import torch
    opsets = [k for k in vars(torch.onnx) if k.startswith("symbolic_opset")]
    if not opsets:
        raise RuntimeError("Could not find supported ONNX opsets in torch.onnx.")
    max_opset = max(int(k[14:]) for k in opsets)
    return max_opset - 1


class UltralyticsExporter(BaseExporter):
    """Ultralytics YOLO model ONNX exporter

    Supports four tasks: detect / segment / classify / pose.
    Compatible with YOLOv8 / YOLO11 and future versions.

    Args:
        model_name: Model name, e.g., "yolov8s", "yolov8s-seg", "yolo11n"
        opset: ONNX opset version (defaults to the second-to-latest)
    """

    def __init__(self, model_name: str, opset: Optional[int] = None):
        super().__init__(model_name, opset or get_latest_opset())
        self._task = self._infer_task(model_name)

    @staticmethod
    def _infer_task(model_name: str) -> str:
        name = model_name.lower()
        if "-seg" in name:
            return "segment"
        if "-cls" in name:
            return "classify"
        if "-pose" in name:
            return "pose"
        return "detect"

    def export_onnx(
        self,
        output_path: str,
        img_size: int = 640,
        half: bool = False,
        do_validation: bool = True,
    ) -> str:
        """Export ONNX model

        Uses Ultralytics' YOLO().export() interface to complete the export.
        This method wraps model loading, format conversion, and opset configuration.

        Args:
            output_path: ONNX save path (used only for renaming/moving; actual export is done by Ultralytics)
            img_size: Input image size
            half: Whether to export FP16 half-precision ONNX
            do_validation: Whether to automatically run ONNX validation + PT output comparison

        Returns:
            Absolute path to the ONNX file
        """
        import torch
        from ultralytics import YOLO

        print(f"[Ultralytics] Loading {self.model_name} ...")
        model = YOLO(self.model_name)

        print(f"[Ultralytics] Exporting to ONNX (opset={self.opset}, imgsz={img_size})")
        model.export(
            format="onnx",
            opset=self.opset,
            imgsz=img_size,
            half=half,
        )

        # Ultralytics generates .onnx at the model's default location; move/rename to target path
        import os
        import shutil

        src_name = self.model_name.replace(".pt", "") + ".onnx"
        if "/" in src_name:
            src_name = src_name.split("/")[-1]
        if os.path.exists(src_name):
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(src_name, output_path)
            print(f"[Ultralytics] ONNX saved to {output_path}")
        else:
            print(f"[Ultralytics] ONNX export completed (expected at {src_name})")

        # Auto-validation
        if do_validation and os.path.exists(output_path):
            check_onnx(output_path)
            try:
                # PT vs ONNX output comparison (using the raw PyTorch model)
                dummy_input = torch.randn(1, 3, img_size, img_size)
                compare_output(
                    onnx_path=output_path,
                    input_tensor=dummy_input,
                    torch_model=model.model,
                )
            except Exception as e:
                print(f"[Ultralytics] PT vs ONNX comparison skipped: {e}")

        return output_path


# ==================== CLI ====================


def parse_opt():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO model ONNX export")
    parser.add_argument(
        "model",
        type=str,
        help="Model name, e.g., yolov8s, yolov8s-seg, yolo11n",
    )
    parser.add_argument(
        "--opset", type=int, default=None,
        help="ONNX opset version (defaults to second-to-latest)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="ONNX save path (defaults to current directory)",
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="Input image size",
    )
    return parser.parse_args()


def main():
    args = parse_opt()
    save_path = args.save or f"{args.model.replace('.pt', '')}.onnx"

    exporter = UltralyticsExporter(args.model, opset=args.opset)
    exporter.export_onnx(output_path=save_path, img_size=args.img_size)


if __name__ == "__main__":
    import torch
    main()
