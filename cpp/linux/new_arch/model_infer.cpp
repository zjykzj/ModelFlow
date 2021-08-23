//
// Created by zj on 2021/8/23.
//

#include "model_infer.h"


bool ModelInfer::create(const char *model_path) {
    std::cout << "ModelInfer::create()" << std::endl;
    return true;
}

bool ModelInfer::release() {
    std::cout << "ModelInfer::release()" << std::endl;
    return true;
}

bool ModelInfer::infer(const cv::Mat &img, std::vector<float> &output_values) {
    std::cout << "ModelInfer::infer()" << std::endl;
    return true;
}
