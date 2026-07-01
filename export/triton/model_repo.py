# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : model_repo.py
@Author  : zj
@Description: Triton model repository directory structure generator

Automatically creates a Triton-compliant model repository directory structure
and places model files (.onnx / .engine) into the correct locations.

Usage:
    >>> from export.triton import ModelRepoBuilder
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
    """Triton model repository builder

    Args:
        repo_root: Root directory of the model repository
    """

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    @staticmethod
    def build_model_name(task: str, dataset: str, architecture: str, backend: str) -> str:
        """Generate a model name following the naming convention

        Args:
            task: Task type ("Detect", "Classify", "Segment", "Pose")
            dataset: Dataset ("COCO", "ImageNet", "Custom")
            architecture: Model architecture ("YOLOv8s", "EfficientNetB0")
            backend: Backend ("ONNX", "TRT")

        Returns:
            Model name, e.g. "Detect_COCO_YOLOv8s_TRT"
        """
        return f"{task}_{dataset}_{architecture}_{backend}"

    @staticmethod
    def _infer_backend(model_file: str) -> str:
        """Infer backend type from model file extension"""
        ext = Path(model_file).suffix.lower()
        if ext == ".onnx":
            return "onnxruntime"
        elif ext in (".engine", ".plan"):
            return "tensorrt"
        else:
            raise ValueError(f"Cannot infer backend from file extension: {ext}")

    @staticmethod
    def _model_filename(backend: str) -> str:
        """Return the model filename expected by Triton for the given backend"""
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
        """Deploy a model file into the Triton repository

        Args:
            model_name: Model name (also used as the directory name)
            model_file: Path to the model file (.onnx or .engine)
            backend: Backend type (inferred from file extension by default)
            version: Model version number
            overwrite: Whether to overwrite an existing file

        Returns:
            Path to the model within the repository
        """
        model_file = os.path.abspath(model_file)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        if backend is None:
            backend = self._infer_backend(model_file)

        # Target path: repo_root/model_name/version/model.{onnx|plan}
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
        """List all models in the repository"""
        if not self.repo_root.exists():
            return []
        return [
            d for d in self.repo_root.iterdir()
            if d.is_dir() and (d / "config.pbtxt").exists()
        ]
