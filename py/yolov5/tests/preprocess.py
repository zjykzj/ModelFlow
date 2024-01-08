# -*- coding: utf-8 -*-

"""
@date: 2024/1/8 下午9:45
@file: preprocess.py
@author: zj
@description: 
"""

import cv2

from py.yolov5.yolov5_util import letterbox


def test_letterbox():
    im0 = cv2.imread("../../../assets/bus.jpg")
    img_size = (640, 640)
    stride = 32

    # 正常模式
    auto = False
    scaleFill = False
    im1 = letterbox(im0, img_size, stride=stride, auto=auto, scaleFill=scaleFill)[0]

    # auto模式
    auto = True
    scaleFill = False
    im2 = letterbox(im0, img_size, stride=stride, auto=auto, scaleFill=scaleFill)[0]

    # scaleFill模式
    auto = False
    scaleFill = True
    im3 = letterbox(im0, img_size, stride=stride, auto=auto, scaleFill=scaleFill)[0]

    cv2.imshow("im0", im0)
    cv2.imshow("im1", im1)
    cv2.imshow("im2", im2)
    cv2.imshow("im3", im3)
    cv2.waitKey(0)

    cv2.imwrite("letterbox_im0.jpg", im0)
    cv2.imwrite("letterbox_im1.jpg", im1)
    cv2.imwrite("letterbox_im2.jpg", im2)
    cv2.imwrite("letterbox_im3.jpg", im3)


if __name__ == '__main__':
    test_letterbox()
