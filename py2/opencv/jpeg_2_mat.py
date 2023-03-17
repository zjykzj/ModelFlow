# -*- coding: utf-8 -*-

"""
@date: 2023/3/17 下午5:51
@file: jpeg_2_mat.py
@author: zj
@description: 
"""

import cv2

import numpy as np


def jpeg2mat():
    with open('../../assets/bus.jpg', 'rb') as f:
        jpeg_data = np.asarray(bytearray(f.read()), dtype="uint8")

    src_img = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)

    cv2.imshow("src_img", src_img)
    cv2.waitKey(0)


def mat2jpeg():
    img = cv2.imread('../../assets/bus.jpg')
    cv2.imshow("src_img", img)
    cv2.waitKey(0)

    img_encode = cv2.imencode('.jpg', img)[1]
    with open('./bus_test.jpg', 'wb') as f:
        f.write(img_encode)


if __name__ == '__main__':
    # jpeg2mat()
    mat2jpeg()
