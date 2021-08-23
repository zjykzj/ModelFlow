//
// Created by zj on 2021/8/19.
//

#ifndef ONNX_TO_ONNX_RUNTIME_INFERENGINE_H
#define ONNX_TO_ONNX_RUNTIME_INFERENGINE_H

#include <onnxruntime_cxx_api.h>
#include <cassert>
#include <common_operation.h>
#include "opencv2/opencv.hpp"

/**
 * refer to
 * onnxruntime/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
 * https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
 * ONNX-Runtime-Inference/src/inference.cpp
 * https://github.com/leimao/ONNX-Runtime-Inference/blob/main/src/inference.cpp
 */
class ONNXInfer {

public:

    ONNXInfer();

    bool create(const char *model_path);

    bool release();

    void print_input_info();

    void print_output_info();

    void infer(const cv::Mat &img, std::vector<float> &output_values);

    static const size_t IMAGE_WIDTH = 224;
    static const size_t IMAGE_HEIGHT = 224;
    static const size_t IMAGE_CHANNEL = 3;
    // simplify ... using known dim values to calculate size
    // use OrtGetTensorShapeElementCount() to get official size!
    const static size_t input_tensor_size = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL;

private:

    // initialize environment...one environment per process
    // environment maintains thread pools and other state info
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "infer"};

    // initialize session options if needed
    Ort::SessionOptions session_options;

    Ort::Session session{nullptr};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // simplify... this model has only 1 input node {1, 3, 224, 224}.
    // Otherwise need vector<vector<>>
    std::vector<int64_t> input_node_dims;
    std::vector<const char *> input_node_names{nullptr};
    Ort::Value input_tensor{nullptr};

    // simplify... this model has only 1 output node {1, N}.
    // Otherwise need vector<vector<>>
    std::vector<int64_t> output_node_dims;
    std::vector<const char *> output_node_names{nullptr};
    Ort::Value output_tensor{nullptr};

    // create input tensor object from data values
    std::vector<float> input_tensor_values = std::vector<float>(input_tensor_size);
    // create output tensor object
    std::vector<float> output_tensor_values;
};


#endif //ONNX_TO_ONNX_RUNTIME_INFERENGINE_H
