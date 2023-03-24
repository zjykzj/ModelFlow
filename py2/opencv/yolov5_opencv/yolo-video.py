# -*- coding: utf-8 -*-

"""
@date: 2023/3/24 下午2:28
@file: yolo-test.py
@author: zj
@description: See https://github.com/doleron/yolov5-opencv-cpp-python/blob/main/python/yolo.py
"""

import os
import cv2
import time
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 OpenCV Video Test.")
    parser.add_argument('-v', '--video', metavar='VIDEO', type=str, default='../../../assets/sample.mp4')
    parser.add_argument('-m', '--model', metavar='MODEL', type=str, default='../../../assets/yolov5n.onnx')
    parser.add_argument('-cls', '--classes', metavar='CLASSES', type=str, default="../../../assets/coco.names")
    parser.add_argument('-o', '--output', metavar='OUTPUT', type=str, default="../../../assets/yolov5-opencv-det.jpg")
    parser.add_argument('--is_cuda', action='store_true', default=False)

    args = parser.parse_args()
    print("args:", args)
    return args


def load_data(image, img_sz=640):
    def format_yolov5(frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    input_image = format_yolov5(image)
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (img_sz, img_sz), swapRB=True, crop=False)
    return input_image, blob


def load_model(model_path, is_cuda=False):
    assert os.path.isfile(model_path), model_path

    model = cv2.dnn.readNet(model_path)
    if is_cuda:
        print("Attempt to use CUDA")
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
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


def draw_results(image, class_path, result_class_ids, result_confidences, result_boxes):
    with open(class_path, "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    for i in range(len(result_class_ids)):
        box = result_boxes[i]
        class_id = result_class_ids[i]

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, f'{class_list[class_id]} {result_confidences[i]:.3f}', (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

    return image


def main():
    args = parse_args()

    model = load_model(args.model, is_cuda=args.is_cuda)
    capture = cv2.VideoCapture(args.video)

    start = time.time_ns()
    frame_count = 0
    total_frames = 0
    fps = -1
    while True:
        _, frame = capture.read()
        if frame is None:
            print("End of stream")
            break

        input_image, blob = load_data(frame)
        model.setInput(blob)
        predictions = model.forward()
        result_class_ids, result_confidences, result_boxes = post_process(predictions[0], input_image.shape)
        draw_image = draw_results(frame, args.classes, result_class_ids, result_confidences, result_boxes, args.output)

        frame_count += 1
        total_frames += 1
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(draw_image, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("output", draw_image)
        if cv2.waitKey(1) > -1:
            print("finished by user")
            break
    print("Total frames: " + str(total_frames))


if __name__ == '__main__':
    main()
