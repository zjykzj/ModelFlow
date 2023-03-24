# -*- coding: utf-8 -*-

"""
@date: 2023/3/24 下午2:28
@file: yolo-test.py
@author: zj
@description: See https://github.com/doleron/yolov5-opencv-cpp-python/blob/main/python/yolo.py
"""

import os
import cv2
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 OpenCV　Image Test.")
    parser.add_argument('-i', '--img', metavar='IMG', type=str, default='../../../assets/bus.jpg')
    parser.add_argument('-m', '--model', metavar='MODEL', type=str, default='../../../assets/yolov5n.onnx')
    parser.add_argument('-cls', '--classes', metavar='CLASSES', type=str, default="../../../assets/coco.names")

    parser.add_argument('-o', '--output', metavar='OUTPUT', type=str, default="../../../assets/yolov5-opencv-det.jpg")
    args = parser.parse_args()
    print("args:", args)
    return args


def load_data(img_path, img_sz=640):
    assert os.path.isfile(img_path), img_path
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    def format_yolov5(frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    input_image = format_yolov5(image)
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (img_sz, img_sz), swapRB=True, crop=False)
    return image, input_image, blob


def load_model(model_path):
    assert os.path.isfile(model_path), model_path

    model = cv2.dnn.readNet(model_path)
    return model


def post_process(output_data, img_shape, img_sz=640, box_num=25200, conf_th=0.4, cls_th=0.25,
                 score_th=0.25, nms_th=0.45):
    image_width, image_height = img_shape[:2]
    x_factor = image_width / img_sz
    y_factor = image_height / img_sz

    class_ids = []
    confidences = []
    boxes = []
    for r in range(box_num):
        row = output_data[r]
        confidence = row[4]
        if confidence >= conf_th:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > cls_th):
                confidences.append(confidence)
                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_th, nms_th)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def show_results(image, class_path, result_class_ids, result_confidences, result_boxes, output_path):
    with open(class_path, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    for i in range(len(result_class_ids)):
        box = result_boxes[i]
        class_id = result_class_ids[i]

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, f'{class_list[class_id]} {result_confidences[i]:.3f}', (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

    cv2.imwrite(output_path, image)
    cv2.imshow("output", image)
    cv2.waitKey()


def main():
    args = parse_args()

    image, input_image, blob = load_data(args.img)
    model = load_model(args.model)

    model.setInput(blob)
    predictions = model.forward()
    result_class_ids, result_confidences, result_boxes = post_process(predictions[0], input_image.shape)

    show_results(image, args.classes, result_class_ids, result_confidences, result_boxes, args.output)


if __name__ == '__main__':
    main()
