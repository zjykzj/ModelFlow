//#include <iostream>
//
//int main() {
//    std::cout << "Hello, World!" << std::endl;
//    return 0;
//}

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocv_common.hpp"

#include <inference_engine.hpp>
#include <iterator>
#include <string>

#include "source/openvino_infer.h"
#include "image_process.h"
#include "source/infer_engine.h"

using namespace InferenceEngine;


/**
* @brief Define names based depends on Unicode path support
*/

#define tcout                  std::cout
#define file_name_t            std::string
#define imread_t               cv::imread

int main(int argc, char *argv[]) {
    // ------------------------------ Parsing and validation of input arguments
    // ---------------------------------
    if (argc != 4) {
        tcout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name>" << std::endl;
        return EXIT_FAILURE;
    }

    const file_name_t input_model{argv[1]};
    const file_name_t input_image_path{argv[2]};
    const std::string device_name{argv[3]};

    clock_t start, end, clock_img_read, clock_img_preprocess, clock_model_create, clock_model_infer;

    start = clock();
    cv::Mat img;
    read_image(input_image_path.c_str(), img, true);
    print_image_info(img);
    clock_img_read = clock();

    cv::Mat preprocess_img;
    InferEngine::preprocess(img, preprocess_img);
    clock_img_preprocess = clock();

    auto model = new InferEngine();
    model->create(input_model.c_str(), device_name.c_str());
    clock_model_create = clock();

    std::vector<float> output_values;
    model->infer(input_image_path.c_str(), preprocess_img, output_values);
    model->release();
    clock_model_infer = clock();

//    std::vector<size_t> output_idxes;
//    std::vector<float> output_probes(output_values.size());;
//    InferEngine::postprocess(output_values, output_idxes, output_probes);
//
//    for (int i = 0; i < 5; i++) {
//        size_t index = output_idxes.at(i);
//        printf("output idx: %zu, output value: %f, output probes: %f\n",
//               index, output_values.at(index), output_probes.at(index));
//    }

    end = clock();
    std::cout << "image read: " << (double) (clock_img_read - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "image preprocess: " << (double) (clock_img_preprocess - clock_img_read) / CLOCKS_PER_SEC << std::endl;
    std::cout << "model create: " << (double) (clock_model_create - clock_img_preprocess) / CLOCKS_PER_SEC << std::endl;
    std::cout << "model infer: " << (double) (clock_model_infer - clock_model_create) / CLOCKS_PER_SEC << std::endl;
    std::cout << "post process: " << (double) (end - clock_model_infer) / CLOCKS_PER_SEC << std::endl;
    std::cout << "total infer: " << (double) (end - start) / CLOCKS_PER_SEC << std::endl;


//    cv::Mat image = imread_t(input_image_path);
//    //            cv::Mat image = imread_t(input_image_path);
//    //            std::cout << image.rows << " " << image.cols << std::endl;
//    // 格式转换
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//    // 零填充
//    padding(image);
//    //            std::cout << image.rows << " " << image.cols << std::endl;
//    //        cv::imwrite("./padding.jpg", image);
//    // 宽高缩放
//    //        cv::resize(image, image, cv::Size(image.rows, image.cols));
//    cv::resize(image, image, cv::Size(224, 224));
//    //            std::cout << image.rows << " " << image.cols << std::endl;
//    // 设置数据类型
//    image.convertTo(image, CV_32F);
//    // 数值缩放
//    image = image / 255.0;
//    // 标准化
//    cv::subtract(image, cv::Scalar(0.45, 0.45, 0.45), image);
//    cv::divide(image, cv::Scalar(0.225, 0.225, 0.225), image);
//
//    auto model = OpenVINOInfer();
//    model.create(input_model.c_str(), device_name.c_str());
//    std::vector<float> output_values;
//    model.infer(input_image_path.c_str(), image, output_values);

    return EXIT_SUCCESS;
}
