//
// Created by zj on 2021/8/23.
//

#ifndef ONNX_TO_ONNXRUNTIME_INFER_ENGINE_H
#define ONNX_TO_ONNXRUNTIME_INFER_ENGINE_H


#include "iostream"
#include "opencv2/opencv.hpp"
#include "onnx_infer.h"

class InferEngine {

public:

    virtual bool create(const char *model_path);

    virtual bool release();

    static bool preprocess(const cv::Mat &src, cv::Mat &dst);

    virtual bool infer(const cv::Mat &img, std::vector<float> &output_values, std::vector<size_t> &output_idxes);

    static bool probes(std::vector<float> &values);

private:

    ONNXInfer *model;

};


#endif //ONNX_TO_ONNXRUNTIME_INFER_ENGINE_H
