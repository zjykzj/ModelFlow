//
// Created by zj on 2022/2/16.
//


/**
 * 使用MNN预处理库进行图像缩放，使用stb库进行读取和保存
 */

#include "iostream"
#include "vector"
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

int main() {
    int width, height, channel;
    unsigned char *data = stbi_load("../assets/lena.jpg", &width, &height, &channel, 0);
    std::cout << "width: " << width << " height: " << height << " channel: " << channel << std::endl;

    int size_w = 400;
    int size_h = 400;

    // 图像预处理
    MNN::CV::Matrix trans;
    // Set transform, from dst scale to src, the ways below are both ok
    trans.setScale((float) width / size_w, (float) height / size_h);
//    trans.setScale((float) (width - 1) / (size_w - 1), (float) (height - 1) / (size_h - 1));

    MNN::CV::ImageProcess::Config cv_config;
    cv_config.filterType = MNN::CV::BILINEAR;
    float means[3] = {0.f, 0.f, 0.f};
    float normals[3] = {1.f, 1.f, 1.f};
    // 归一化操作
    ::memcpy(cv_config.mean, means, sizeof(means));
    ::memcpy(cv_config.normal, normals, sizeof(normals));
    // 指定预处理前后图像格式 当前不进行数据格式转换
    cv_config.sourceFormat = MNN::CV::RGB;
    cv_config.destFormat = MNN::CV::RGB;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(cv_config));

    std::vector<uint8_t> dst_data(1 * size_w * size_h * channel);
    std::shared_ptr<MNN::Tensor> inputUser(
            MNN::Tensor::create<uint8_t>(std::vector<int>{1, size_h, size_w, channel}, dst_data.data(),
                                         MNN::Tensor::TENSORFLOW)
    );
    pretreat->setMatrix(trans);
    pretreat->convert((uint8_t *) data, width, height, 0,
                      inputUser->host<uint8_t>(),
                      size_w, size_h, channel, 0, inputUser->getType());
    std::cout << inputUser->getDimensionType() << std::endl;

    stbi_write_jpg("../assets/mnn_resize.jpg", size_w, size_h, channel, inputUser->host<uint8_t>(), 100);
    stbi_image_free(data);

    std::cout << "Norm Compare" << std::endl;
    return 0;
}