# -*- coding: utf-8 -*-

"""
@Time    : 2024/9/7 15:31
@File    : runtime_bakcend.py
@Author  : zj
@Description: 
"""

import os

import numpy as np
from numpy import ndarray


class BackendRuntime:

    def __init__(self, weight: str = 'yolov5s.onnx', providers=None):
        super().__init__()
        self.load_onnx(weight, providers)

    def load_onnx(self, weight: str, providers=None):
        assert os.path.isfile(weight), weight
        if providers is None:
            providers = ['CPUExecutionProvider']

        print(f'Loading {weight} for ONNX Runtime inference...')
        import onnxruntime
        session = onnxruntime.InferenceSession(weight, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        metadata = session.get_modelmeta().custom_metadata_map  # metadata
        print(f"metadata: {metadata}")

        self.session = session
        self.output_names = output_names
        self.dtype = np.float32
        print(f"Init Done. Work with {self.dtype}")

    def __call__(self, im: ndarray):
        im = im.astype(self.dtype)
        preds = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        return preds
