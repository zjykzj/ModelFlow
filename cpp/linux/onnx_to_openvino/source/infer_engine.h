//
// Created by zj on 2021/8/24.
//

#ifndef ONNX_TO_OPENVINO_INFER_ENGINE_H
#define ONNX_TO_OPENVINO_INFER_ENGINE_H

#include "openvino_infer.h"

class InferEngine {

public:

    virtual bool create(const char *model_path, const char *device_name);

    virtual bool release();

    static bool preprocess(const cv::Mat &src, cv::Mat &dst);

    virtual bool infer(const cv::Mat &img, std::vector<float> &output_values);

    static bool
    postprocess(const std::vector<float> &output_values, std::vector<size_t> &output_idxes, std::vector<float> &probes);

private:

    OpenVINOInfer *model{};

};


#endif //ONNX_TO_OPENVINO_INFER_ENGINE_H
