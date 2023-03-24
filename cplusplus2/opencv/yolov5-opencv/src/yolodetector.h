//
// Created by zj on 23-3-24.
//

#ifndef YOLOV5_OPENCV_OPENCV_YOLOV5_OPENCV_SRC_YOLODETECTOR_H_
#define YOLOV5_OPENCV_OPENCV_YOLOV5_OPENCV_SRC_YOLODETECTOR_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

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
  static cv::Mat FormatYOLOv5(const cv::Mat &source);

  std::shared_ptr<cv::dnn::Net> net_ = nullptr;
};

#endif //YOLOV5_OPENCV_OPENCV_YOLOV5_OPENCV_SRC_YOLODETECTOR_H_
