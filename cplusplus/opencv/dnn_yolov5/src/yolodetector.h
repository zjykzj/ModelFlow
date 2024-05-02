//
// Created by zj on 23-3-24.
//

#ifndef YOLOV5_OPENCV_OPENCV_YOLOV5_OPENCV_SRC_YOLODETECTOR_H_
#define YOLOV5_OPENCV_OPENCV_YOLOV5_OPENCV_SRC_YOLODETECTOR_H_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.25;

struct BoxInfo {
    int class_id;
    float confidence;
    cv::Rect box;
};

class YOLODetector {
   public:
    YOLODetector() = default;
    ~YOLODetector() = default;

    int Init(const std::string &model_path, bool is_cuda);

    int Detect(const cv::Mat &image, const std::vector<std::string> &class_names, std::vector<BoxInfo> &output);

   private:
    static std::tuple<cv::Mat, int, int> FormatYOLOv5(const cv::Mat &source);

    std::shared_ptr<cv::dnn::Net> net_ = nullptr;
};

#endif  // YOLOV5_OPENCV_OPENCV_YOLOV5_OPENCV_SRC_YOLODETECTOR_H_
