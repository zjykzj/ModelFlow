//
// Created by zj on 23-3-24.
//

#include "yolodetector.h"

std::tuple<cv::Mat, int, int> YOLODetector::FormatYOLOv5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);

    // 计算居中偏移量
    int offset_x = (_max - col) / 2;
    int offset_y = (_max - row) / 2;
    // 将输入图像居中放置在结果图像中
    source.copyTo(result(cv::Rect(offset_x, offset_y, col, row)));

    return std::make_tuple(result, offset_x, offset_y);
}

int YOLODetector::Init(const std::string &model_path, bool is_cuda) {
    this->net_ = std::make_shared<cv::dnn::Net>(cv::dnn::readNet(model_path));
    if (is_cuda) {
        std::cout << "Attempt to use CUDA\n";
        this->net_->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net_->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        std::cout << "Running on CPU\n";
        this->net_->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        this->net_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    return 0;
}

int YOLODetector::Detect(const cv::Mat &image, const std::vector<std::string> &class_names,
                         std::vector<BoxInfo> &output) {
    cv::Mat letterBox;
    int offset_x, offset_y;
    std::tie(letterBox, offset_x, offset_y) = this->FormatYOLOv5(image);

    cv::Mat blob;
    cv::dnn::blobFromImage(letterBox, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    this->net_->setInput(blob);
    std::vector<cv::Mat> outputs;
    this->net_->forward(outputs, this->net_->getUnconnectedOutLayersNames());

    float x_factor = float(letterBox.cols) / INPUT_WIDTH;
    float y_factor = float(letterBox.rows) / INPUT_HEIGHT;
    const int dimensions = 85;
    const int rows = 25200;

    float *data = (float *)outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < rows; ++i) {
        float obj_conf = data[4];
        if (obj_conf >= CONFIDENCE_THRESHOLD) {
            float *classes_scores = data + 5;
            cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id);

            float conf = obj_conf * max_class_score;
            if (conf > CONFIDENCE_THRESHOLD) {
                confidences.push_back(conf);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor) - offset_x;
                int top = int((y - 0.5 * h) * y_factor) - offset_y;
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.emplace_back(left, top, width, height);
            }
        }
        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int idx : nms_result) {
        BoxInfo result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    return 0;
}
