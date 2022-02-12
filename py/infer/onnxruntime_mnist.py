# -*- coding: utf-8 -*-

"""
@date: 2022/2/12 下午2:51
@file: onnxruntime_mnist.py
@author: zj
@description: 
"""

import onnxruntime

import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def load_data(img_path):
    img = Image.open(img_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img_y = transform(img)

    # resize = transforms.Resize([224, 224])
    # img = resize(img)

    # img_ycbcr = img.convert('YCbCr')
    # img_y, img_cb, img_cr = img_ycbcr.split()

    # to_tensor = transforms.ToTensor()
    # img_y = to_tensor(img_y)
    # img_y.unsqueeze_(0)

    return img_y.unsqueeze_(0)


def load_model(onnx_path):
    ort_session = onnxruntime.InferenceSession(onnx_path)
    return ort_session


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def main():
    img_y = load_data("../assets/mnist_0.png")
    ort_session = load_model('../assets/mnist_cnn.onnx')

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    print(img_out_y)
    # compute probs
    probs = softmax(img_out_y)
    print(probs)
    # compute target
    targets = np.argmax(probs, axis=1)
    print(targets)


if __name__ == '__main__':
    main()
