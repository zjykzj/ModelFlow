//
// Created by zj on 2021/8/24.
//

#ifndef ONNX_TO_OPENVINO_OPENVINO_INFER_H
#define ONNX_TO_OPENVINO_OPENVINO_INFER_H

#include <inference_engine.hpp>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <cassert>
#include <common_operation.h>
#include "opencv2/opencv.hpp"


using namespace InferenceEngine;

class OpenVINOInfer {


public:

    OpenVINOInfer();

    bool create(const char *model_path, const char *device_name);

    bool release();

    void print_input_info();

    void print_output_info();

    bool infer(const char *img_path, const cv::Mat &img, std::vector<float> &output_values);

    static const size_t IMAGE_WIDTH = 224;
    static const size_t IMAGE_HEIGHT = 224;
    static const size_t IMAGE_CHANNEL = 3;
    // simplify ... using known dim values to calculate size
    // use OrtGetTensorShapeElementCount() to get official size!
    const static size_t input_tensor_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL;

private:

    // Step 1. Initialize inference engine core
    Core ie;

    CNNNetwork network;

    // Step 4. Loading a model to the device
    ExecutableNetwork executable_network;

    InferRequest infer_request;

    std::string input_name;
    std::string output_name;
};


#endif //ONNX_TO_OPENVINO_OPENVINO_INFER_H
