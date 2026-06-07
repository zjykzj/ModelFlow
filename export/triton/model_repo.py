# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : model_repo.py
@Author  : zj
@Description: Triton 模型仓库目录结构生成器

自动创建符合 Triton 规范的模型仓库目录结构，
并将模型文件（.onnx / .engine）放置到正确位置。

用法：
    >>> from export2.triton import ModelRepoBuilder
    >>> builder = ModelRepoBuilder("models/triton/")
    >>> builder.deploy(
    ...     model_name="Detect_COCO_YOLOv8s_TRT",
    ...     model_file="yolov8s_fp16.engine",
    ...     backend="tensorrt",
    ...     version=1,
    ... )
"""

import os
import shutil
from pathlib import Path
from typing import Optional


class ModelRepoBuilder:
    """Triton 模型仓库构建器

    Args:
        repo_root: 模型仓库根目录
    """

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    @staticmethod
    def build_model_name(task: str, dataset: str, architecture: str, backend: str) -> str:
        """按规范生成模型名称

        Args:
            task: 任务类型 ("Detect", "Classify", "Segment", "Pose")
            dataset: 数据集 ("COCO", "ImageNet", "Custom")
            architecture: 模型架构 ("YOLOv8s", "EfficientNetB0")
            backend: 后端 ("ONNX", "TRT")

        Returns:
            模型名称，如 "Detect_COCO_YOLOv8s_TRT"
        """
        return f"{task}_{dataset}_{architecture}_{backend}"

    @staticmethod
    def _infer_backend(model_file: str) -> str:
        """根据模型文件后缀推断后端类型"""
        ext = Path(model_file).suffix.lower()
        if ext == ".onnx":
            return "onnxruntime"
        elif ext in (".engine", ".plan"):
            return "tensorrt"
        else:
            raise ValueError(f"Cannot infer backend from file extension: {ext}")

    @staticmethod
    def _model_filename(backend: str) -> str:
        """根据后端返回 Triton 期望的模型文件名"""
        if backend in ("onnxruntime", "onnx"):
            return "model.onnx"
        elif backend in ("tensorrt", "trt"):
            return "model.plan"
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def deploy(
        self,
        model_name: str,
        model_file: str,
        backend: Optional[str] = None,
        version: int = 1,
        overwrite: bool = True,
    ) -> str:
        """将模型文件部署到 Triton 仓库

        Args:
            model_name: 模型名称（同时作为目录名）
            model_file: 模型文件路径（.onnx 或 .engine）
            backend: 后端类型（默认根据文件后缀推断）
            version: 模型版本号
            overwrite: 是否覆盖已存在的文件

        Returns:
            模型在仓库中的路径
        """
        model_file = os.path.abspath(model_file)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        if backend is None:
            backend = self._infer_backend(model_file)

        # 目标路径：repo_root/model_name/version/model.{onnx|plan}
        model_dir = self.repo_root / model_name / str(version)
        self._ensure_dir(model_dir)

        target_file = model_dir / self._model_filename(backend)

        if target_file.exists() and not overwrite:
            print(f"[Triton] ⚠️  {target_file} exists, skipping (overwrite=False)")
        else:
            shutil.copy2(model_file, target_file)
            size_mb = os.path.getsize(target_file) / 1024 / 1024
            print(f"[Triton] ✅ Copied {model_file} -> {target_file} ({size_mb:.2f} MB)")

        return str(self.repo_root / model_name)

    def list_models(self) -> list:
        """列出仓库中的所有模型"""
        if not self.repo_root.exists():
            return []
        return [
            d for d in self.repo_root.iterdir()
            if d.is_dir() and (d / "config.pbtxt").exists()
        ]
