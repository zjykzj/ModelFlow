# -*- coding: utf-8 -*-

"""
@date: 2021/7/3 下午3:22
@file: torchvision_vs_opencv.py
@author: zj
@description: Compare torchvision.transform and opencv implementation with the same preprocessing operation
"""

import cv2
import time
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms

torchvision.set_image_backend('accimage')


def get_transform():
    transform = transforms.Compose([
        transforms.Resize(shorter_side),
        transforms.CenterCrop((shorter_side, shorter_side)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def torchvision_operation(image, transform):
    image = Image.fromarray(image)

    return transform(image).numpy()


def opencv_operation(image):
    h, w = image.shape[:2]
    if h > w:
        ratio = 1.0 * shorter_side / w

        resize_img = cv2.resize(image, (w * ratio, h * ratio))
        h, w = resize_img.shape[:2]
        crop_img = cv2[int(h / 2 - shorter_side / 2): int(h / 2 + shorter_side / 2)]
    else:
        ratio = 1.0 * 112 / h

        resize_img = cv2.resize(image, (int(w * ratio), int(h * ratio)))
        h, w = resize_img.shape[:2]
        crop_img = resize_img[:, int(w / 2.0 - shorter_side / 2.0): int(w / 2.0 + shorter_side / 2.0)]

    normalized_img = crop_img * 1.0 / 255
    mean_img = normalized_img - MEAN
    std_img = mean_img / STD
    return std_img.transpose(2, 0, 1)


if __name__ == '__main__':
    """
    1. Scale image equally, minimum edge to 112
    2. Crop center size
    3. Scale the value to between (0,1)
    4. Data standardization, mean = (0.45, 0.45, 0.45), STD = (0.225, 0.225, 0.225)
    """
    shorter_side = 112
    MEAN = (0.45, 0.45, 0.45)
    STD = (0.225, 0.225, 0.225)

    transform = get_transform()

    num = 100
    t_time = 0.
    o_time = 0.
    for i in range(num):
        time.sleep(0.1)
        image = np.ones((240, 326, 3)).astype(np.uint8)
        t0 = time.time()
        res_1 = torchvision_operation(image, transform)
        t1 = time.time()
        res_2 = opencv_operation(image)
        t2 = time.time()

        assert np.allclose(res_1, res_2, rtol=1.e-5, atol=1.e-8)

        if i < num / 2:
            continue
        t_time += (t1 - t0)
        o_time += (t2 - t1)

    print('torchvision: ', t_time / num * 2)
    print('opencv: ', o_time / num * 2)
