# -*- coding: utf-8 -*-

"""
@date: 2023/3/17 下午4:43
@file: rotate_crop.py
@author: zj
@description: 
"""

import cv2
import numpy as np

coords = np.array([[43, 394], [188, 388], [264, 911], [35, 906]])


def demo1():
    img = np.zeros((1080, 1920, 3), np.uint8)
    # triangle = np.array([[0, 0], [1500, 800], [500, 400]])

    cv2.fillConvexPoly(img, coords, (255, 255, 255))

    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = '../../assets/bus.jpg'
    src_img = cv2.imread(img_path)

    mask = np.zeros(src_img.shape[:2], dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, coords, (255, 255, 255))

    # dst_img = cv2.bitwise_or(src_img, np.zeros(np.shape(src_img), dtype=np.uint8), mask=mask)
    dst_img = cv2.add(src_img, np.zeros(np.shape(src_img), dtype=np.uint8), mask=mask)

    x, y, w, h = cv2.boundingRect(coords)
    crop_img = dst_img[y:y + h, x:x + w]

    cv2.imshow('mask', mask)
    cv2.imshow('src_img', src_img)
    cv2.imshow('dst_img', dst_img)
    cv2.imshow('crop_img', crop_img)
    cv2.waitKey(0)
