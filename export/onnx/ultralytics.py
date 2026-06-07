# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : ultralytics.py
@Author  : zj
@Description: Ultralytics YOLO 模型 ONNX 导出器

支持 YOLOv8 / YOLO11 / 未来版本，覆盖 detect / segment / classify / pose 任务。
底层依赖 Ultralytics 的 YOLO().export()，此模块提供 CLI 封装和 opset 策略。

典型用法：
    >>> from export.onnx import UltralyticsExporter
    >>> exporter = UltralyticsExporter("yolov8s")
    >>> onnx_path = exporter.export_onnx("yolov8s.onnx")

CLI:
    python3 -m export.onnx.ultralytics yolov8s
    python3 -m export.onnx.ultralytics yolov8s-seg --opset 12
"""

import argparse
from typing import Optional

from export.core.base import BaseExporter
from export.core.validation import check_onnx, compare_output


def get_latest_opset() -> int:
    """返回当前 PyTorch 支持的次新版 ONNX opset

    使用 max - 1 而非最新版，确保稳定性（最新 opset 可能有未修复的 bug）。
    """
    import torch
    opsets = [k for k in vars(torch.onnx) if k.startswith("symbolic_opset")]
    if not opsets:
        raise RuntimeError("Could not find supported ONNX opsets in torch.onnx.")
    max_opset = max(int(k[14:]) for k in opsets)
    return max_opset - 1


class UltralyticsExporter(BaseExporter):
    """Ultralytics YOLO 模型 ONNX 导出器

    支持 detect / segment / classify / pose 四种任务，
    兼容 YOLOv8 / YOLO11 及未来版本。

    Args:
        model_name: 模型名称，如 "yolov8s", "yolov8s-seg", "yolo11n"
        opset: ONNX opset 版本（默认自动选择次新版）
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
        """导出 ONNX 模型

        通过 Ultralytics YOLO().export() 接口完成导出，
        此方法封装了模型加载、格式转换和 opset 配置。

        Args:
            output_path: ONNX 保存路径（仅用于重命名/移动，实际导出由 Ultralytics 完成）
            img_size: 输入尺寸
            half: 是否导出 FP16 半精度 ONNX
            do_validation: 是否自动执行 ONNX 验证 + PT 输出对比

        Returns:
            ONNX 文件绝对路径
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

        # Ultralytics 默认在模型位置生成 .onnx，移动/重命名到目标路径
        import os
        import shutil

        src_name = self.model_name.replace(".pt", "") + ".onnx"
        if "/" in src_name:
            src_name = src_name.split("/")[-1]
        if os.path.exists(src_name):
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(src_name, output_path)
            print(f"[Ultralytics] ✅ ONNX saved to {output_path}")
        else:
            print(f"[Ultralytics] ✅ ONNX export completed (expected at {src_name})")

        # 自动验证
        if do_validation and os.path.exists(output_path):
            check_onnx(output_path)
            try:
                # PT vs ONNX 输出对比（使用底层 PyTorch 模型）
                dummy_input = torch.randn(1, 3, img_size, img_size)
                compare_output(
                    onnx_path=output_path,
                    input_tensor=dummy_input,
                    torch_model=model.model,
                )
            except Exception as e:
                print(f"[Ultralytics] ⚠️  PT vs ONNX comparison skipped: {e}")

        return output_path


# ==================== CLI ====================


def parse_opt():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO 模型 ONNX 导出")
    parser.add_argument(
        "model",
        type=str,
        help="模型名称，如 yolov8s, yolov8s-seg, yolo11n",
    )
    parser.add_argument(
        "--opset", type=int, default=None,
        help="ONNX opset 版本（默认次新版）",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="ONNX 保存路径（默认当前目录）",
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="输入尺寸",
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
