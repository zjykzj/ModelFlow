//
// Created by zj on 2021/8/23.
//

#ifndef NEW_ARCH_MODEL_INFER_BASE_H
#define NEW_ARCH_MODEL_INFER_BASE_H

#include "opencv2/opencv.hpp"

class ModelInfer {

public:

    virtual bool create(const char* model_path);

    virtual bool release();

    virtual bool infer(const cv::Mat &img, std::vector<float> &output_values);

    static const size_t IMAGE_WIDTH = 224;
    static const size_t IMAGE_HEIGHT = 224;
    static const size_t IMAGE_CHANNEL = 3;

};

#endif //NEW_ARCH_MODEL_INFER_BASE_H
