//
// Created by zj on 2021/8/23.
//

#include "infer_engine.h"
#include "image_process.h"
#include "common_operation.h"

bool InferEngine::create(const char *model_path) {
    model = new ModelInfer();
    return model->create(model_path);
}

bool InferEngine::release() {
    return model->release();
}

bool InferEngine::preprocess(const cv::Mat &src, cv::Mat dst) {
    cv::Mat img = src.clone();

    square_padding(img);
    print_image_info(img);

    cv::Mat resize_img;
    resize(img, resize_img, cv::Size2i(ModelInfer::IMAGE_WIDTH, ModelInfer::IMAGE_HEIGHT));
    print_image_info(resize_img);

    normalize(resize_img, true, cv::Scalar(0.45, 0.45, 0.45), cv::Scalar(0.225, 0.225, 0.225), 255.0);
    print_image_info(resize_img);

    hwc_2_chw(resize_img, dst);

    return true;
}


bool InferEngine::infer(const cv::Mat &img, std::vector<float> &output_values, std::vector<int> &output_idxes) {
    return model->infer(img, output_values);
}


bool InferEngine::probes(std::vector<float> &values) {
    softmax(values);
    return true;
}
