//
// Created by zj on 2021/8/23.
//

#ifndef NEW_ARCH_INFER_ENGINE_H
#define NEW_ARCH_INFER_ENGINE_H

#include "iostream"
#include "opencv2/opencv.hpp"
#include "model_infer.h"

class InferEngine {

public:

    virtual bool create(const char *model_path);

    virtual bool release();

    static bool preprocess(const cv::Mat &src, cv::Mat &dst);

    virtual bool infer(const cv::Mat &img, std::vector<float> &output_values);

    static bool postprocess(const std::vector<float> &output_values, std::vector<size_t> &output_idxes, std::vector<float> &probes);

private:

    ModelInfer *model;

};


#endif //NEW_ARCH_INFER_ENGINE_H
