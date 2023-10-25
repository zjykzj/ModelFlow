# -*- coding: utf-8 -*-

"""
@date: 2023/3/19 下午3:07
@file: object_detect.py
@author: zj
@description: 
"""
import copy

import cv2
import time

import numpy as np


def detect(img, verbose=False, version='v1'):
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if verbose:
        cv2.imshow("gray", gray)

    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.Canny(blur, 50, 150)
    if verbose:
        cv2.imshow('thresh', thresh)

    if 'v1' == version:
        kernel = np.ones((10, 10), np.uint8)
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=6)
        open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=3)
        if verbose:
            cv2.imshow("close", close)
            cv2.imshow('open', open)

        contours, hierarchy = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif 'v2' == version:
        kernel = np.ones((10, 10), np.uint8)
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel, iterations=1)
        close2 = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel, iterations=6)
        if verbose:
            cv2.imshow('close', close)
            cv2.imshow('open', open)
            cv2.imshow('close2', close2)

        contours, hierarchy = cv2.findContours(close2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        raise ValueError("ERROR")

    rect_list = list()
    for cnt in contours:
        # x, y, w, h
        rect = cv2.boundingRect(cnt)
        area = rect[2] * rect[3]

        if area < 70000:
            continue
        if rect[2] > (width * 0.95) or rect[3] > (height * 0.95):
            continue

        rect_list.append(rect)

    if verbose:
        draw_img = copy.deepcopy(img)
    if len(rect_list) == 0:
        dst_rect = [0, 0, width, height]
        if verbose:
            cv2.rectangle(draw_img, [0, 0, width, height], (0, 255, 255), 2, cv2.LINE_8)
    else:
        if len(rect_list) == 1:
            dst_rect = rect_list[0]
            if verbose:
                cv2.rectangle(draw_img, rect_list[0], (0, 255, 255), 2, cv2.LINE_8)
        else:
            rect_array = np.array(rect_list)
            rect_array[:, 2] = rect_array[:, 2] + rect_array[:, 0]
            rect_array[:, 3] = rect_array[:, 3] + rect_array[:, 1]

            min_x1 = np.min(rect_array[:, 0])
            min_y1 = np.min(rect_array[:, 1])
            max_x2 = np.max(rect_array[:, 2])
            max_y2 = np.max(rect_array[:, 3])

            dst_rect = [min_x1, min_y1, max_x2 - min_x1, max_y2 - min_y1]
            if verbose:
                cv2.rectangle(draw_img, (min_x1, min_y1), (max_x2, max_y2), (0, 255, 255), 2, cv2.LINE_8)

    if verbose:
        draw_img = cv2.drawContours(draw_img, contours, -1, (0, 255, 0), 5)
        cv2.imshow("draw_img", draw_img)

    if dst_rect is None:
        raise ValueError('ERROR')

    x1 = dst_rect[0]
    y1 = dst_rect[1]
    x2 = dst_rect[0] + dst_rect[2]
    y2 = dst_rect[1] + dst_rect[3]
    dst_img = img[y1:y2, x1:x2]
    return dst_img


if __name__ == '__main__':
    img_path = './asset/demo.jpg'
    src_img = cv2.imread(img_path)

    start = time.time()
    dst_img = detect(src_img, verbose=True, version='v1')
    # dst_img = detect(src_img, verbose=True, version='v2')
    end = time.time()

    cv2.imshow("src_img", src_img)
    cv2.imshow("dst_img", dst_img)
    cv2.waitKey(0)

    print('done')
