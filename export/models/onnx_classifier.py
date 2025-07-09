# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 19:46
@File    : onnx_classifier.py
@Author  : zj
@Description: 
"""

import numpy as np

import onnx
import onnxruntime as ort


class ONNXClassifier:
    def __init__(self, model_path: str):
        """
        Load an ONNX model for inference and print model metadata.

        Args:
            model_path (str): Path to the .onnx file.
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Load raw ONNX model for metadata
        self.onnx_model = onnx.load(model_path)

        print(f"✅ ONNX model loaded from {model_path}")
        self._print_model_info()

    def _print_model_info(self):
        """Print detailed information about the ONNX model."""
        print("🔍 ONNX Model Info:")
        print(f"  IR Version:       {self.onnx_model.ir_version}")
        print(f"  Opset Version:    {self.onnx_model.opset_import[0].version}")
        print(f"  Producer Name:    {self.onnx_model.producer_name}")
        print(f"  Producer Version: {self.onnx_model.producer_version}")
        print(f"  Input(s):")
        for inp in self.session.get_inputs():
            shape = [dim if dim != -1 else '?' for dim in inp.shape]
            print(f"    {inp.name} | {inp.type} | Shape: {shape} | Data Type: {inp.type}")
        print(f"  Output(s):")
        for out in self.session.get_outputs():
            shape = [dim if dim != -1 else '?' for dim in out.shape]
            print(f"    {out.name} | {out.type} | Shape: {shape} | Data Type: {out.type}")

        # Estimate parameter count (only weights, not dynamic input)
        initializer_count = sum(np.prod(tensor.dims) for tensor in self.onnx_model.graph.initializer)
        print(f"  Approximate Parameter Count: {int(initializer_count):,}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the input data.

        Args:
            input_data (np.ndarray): Input tensor as a NumPy array.

        Returns:
            np.ndarray: Model output.
        """
        outputs = self.session.run(None, {self.input_name: input_data})
        return outputs[0]
