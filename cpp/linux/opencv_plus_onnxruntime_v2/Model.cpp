//
// Created by zj on 2021/5/28.
//

#include "Model.h"

template<typename T>
static void softmax(T &input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

Model::Model(const char *model_path) {
    // initialize session options if needed
    Ort::SessionOptions session_options;
//    session_options.SetIntraOpNumThreads(1);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = new Ort::Session(env,
                                model_path,
                                session_options);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//        std::cout << "adfa  " << input_image_.size() << "  " << input_shape_.size() << std::endl;
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
                                                    input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
                                                     output_shape_.data(), output_shape_.size());
}

std::ptrdiff_t Model::Run() {
//    const char *input_names[] = {"inputs"};
//    const char *output_names[] = {"outputs"};
    const char *input_names[] = {"inputs"};
    const char *output_names[] = {"outputs"};

    session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
//    for (int i = 0; i < numClasses; i++) {
//        printf("Score for class [%d] =  %f\n", i, results_[i]);
//    }

    softmax(results_);

//    for (int i = 0; i < numClasses; i++) {
//        printf("Prob for class [%d] =  %f\n", i, results_[i]);
//    }

    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
}

cv::Mat Model::imagePreprocess(const std::string &imageFilepath) {
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(width_, height_),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);
// Normalization per channel
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
// HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    std::copy(preprocessedImage.begin<float>(), preprocessedImage.end<float>(), input_image_.data());

    return preprocessedImage;
}