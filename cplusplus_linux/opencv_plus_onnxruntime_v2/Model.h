//
// Created by zj on 2021/5/28.
//

#ifndef OPENCV_ONNXRUNTIME_V2_MODEL_H
#define OPENCV_ONNXRUNTIME_V2_MODEL_H

// Refer to
// onnxruntime/samples/c_cxx/MNIST/MNIST.cpp
// https://github.com/microsoft/onnxruntime/blob/master/samples/c_cxx/MNIST/MNIST.cpp

#include <assert.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <valarray>
#include <array>
#include <cmath>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>


// After instantiation, set the input_image_ data to be the HxW pixel image to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
class Model {
public:
    Model(const char *model_path);

    std::ptrdiff_t Run();

    cv::Mat imagePreprocess(const std::string &imageFilepath);

    static constexpr const int channel = 3;
    static constexpr const int numClasses = 10;
    static constexpr const int width_ = 224;
    static constexpr const int height_ = 224;

    std::array<float, channel * width_ * height_> input_image_{};
    std::array<float, numClasses> results_{};
    int64_t result_{0};

private:
    Ort::Env env;
    Ort::Session *session_;

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, channel, width_, height_};

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, numClasses};
};

#endif //OPENCV_ONNXRUNTIME_V2_MODEL_H
