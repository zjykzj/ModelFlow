//
// Created by zj on 2022/2/16.
//

/**
 * 创建相同的数据，分别使用手动归一化和MNN归一化函数进行预处理，比较精度变化
 * 执行数据归一化规则参考MNN实现：dst = (src - mean) * normal
 */

#include "iostream"
#include "vector"
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"

static std::vector<unsigned char> getSourceData(int width, int height, int channel) {
    std::vector<unsigned char> data(width * height * channel);

    int num = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // HWC数据排列格式
            for (int k = 0; k < channel; k++) {
                data[i * width * channel + j * channel + k] = uint8_t(num);
                num += 1;
            }
        }
    }

    return data;
}

int main() {
    int width = 4;
    int height = 2;
    int channel = 3;
    auto src_data = getSourceData(width, height, channel);
    std::cout << "size: " << src_data.size() << std::endl;
    for (auto &item: src_data) {
        std::cout << int(item) << " ";
    }
    std::cout << std::endl;

    MNN::CV::ImageProcess::Config cv_config;
    cv_config.filterType = MNN::CV::BILINEAR;
    float means[3] = {0.6f, 0.5f, 0.4f};
    float normals[3] = {0.3f, 0.2f, 0.1f};
    // 归一化操作
    ::memcpy(cv_config.mean, means, sizeof(means));
    ::memcpy(cv_config.normal, normals, sizeof(normals));
    // 指定预处理前后图像格式 当前不进行数据格式转换
    cv_config.sourceFormat = MNN::CV::RGB;
    cv_config.destFormat = MNN::CV::RGB;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(cv_config));

    std::vector<float> dst_data(1 * width * height * channel);
    std::shared_ptr<MNN::Tensor> inputUser(
            MNN::Tensor::create<float>(std::vector<int>{1, height, width, channel}, dst_data.data(),
                                       MNN::Tensor::TENSORFLOW)
    );
    pretreat->convert((uint8_t *) src_data.data(), width, height, 0,
                      inputUser->host<float>(),
                      width, height, channel, 0, inputUser->getType());
    auto *pt = inputUser->host<float>();
    for (int i = 0; i < dst_data.size(); i++) {
//        std::cout << pt[i] << " ";
        std::cout << dst_data[i] << " ";
    }
    std::cout << std::endl;

    int num = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // HWC数据排列格式
            for (int k = 0; k < channel; k++) {
                auto mean = means[k];
                auto normal = normals[k];

                float s = src_data[i * width * channel + j * channel + k];
                auto t = (s - mean) * normal;
                std::cout << t << " ";
            }
        }
    }
    std::cout << std::endl;

    std::cout << "Norm Compare" << std::endl;
    return 0;
}


