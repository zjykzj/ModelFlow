//
// Created by zj on 2022/2/16.
//

/**
 * 分别使用手动归一化实现和MNN归一化函数进行预处理，比较精度变化
 * MNN数据归一化实现规则：dst = (src - mean) * normal
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

std::vector<float> customNormalize(std::vector<unsigned char> data, float *means, float *normals,
                                   int width, int height, int channel) {
    assert(data.size() == (width * height * channel));

    std::vector<float> dstData(data.size());
    // HWC数据排列格式
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int k = 0; k < channel; k++) {
                auto mean = means[k];
                auto normal = normals[k];

                auto idx = y * width * channel + x * channel + k;
                dstData[idx] = (data[idx] - mean) * normal;
            }
        }
    }

    return dstData;
}

std::vector<float> mnnNormalize(std::vector<unsigned char> data, float *means, float *normals,
                                int width, int height, int channel) {
    assert(data.size() == (width * height * channel));

    std::vector<float> dstData(width * height * channel);
    std::shared_ptr<MNN::Tensor> inputUser(
            MNN::Tensor::create<float>(std::vector<int>{1, height, width, channel}, dstData.data(),
                                       MNN::Tensor::TENSORFLOW)
    );

    MNN::CV::ImageProcess::Config cv_config;
    // 归一化操作
    ::memcpy(cv_config.mean, means, sizeof(means) * channel);
    ::memcpy(cv_config.normal, normals, sizeof(normals) * channel);
    // 指定预处理前后图像格式 当前不进行数据格式转换
    cv_config.sourceFormat = MNN::CV::RGB;
    cv_config.destFormat = MNN::CV::RGB;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(cv_config));
    pretreat->convert((uint8_t *) data.data(), width, height, 0,
                      inputUser->host<float>(),
                      width, height, channel, 0, inputUser->getType());

    return dstData;
}

template<typename T>
void printData(std::vector<T> data, int width, int height, int channel) {
    assert(data.size() == (width * height * channel));

    // HWC数据排列格式
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int k = 0; k < channel; k++) {
                auto idx = y * width * channel + x * channel + k;
                std::cout << (float) data[idx] << "/";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int width = 4;
    int height = 2;
    int channel = 3;
    auto src_data = getSourceData(width, height, channel);
    std::cout << "width: " << width << " height: " << height << " channel: " << channel << " size: " << src_data.size()
              << std::endl;
    printData(src_data, width, height, channel);

    float means[3] = {0.6f, 0.5f, 0.4f};
    float normals[3] = {0.3f, 0.2f, 0.1f};

    std::cout << "Custom Normalize..." << std::endl;
    auto customData = customNormalize(src_data, means, normals, width, height, channel);
    printData(customData, width, height, channel);

    std::cout << "MNN Normalize..." << std::endl;
    auto mnnData = mnnNormalize(src_data, means, normals, width, height, channel);
    printData(mnnData, width, height, channel);

    std::cout << "Norm Compare" << std::endl;
    return 0;
}