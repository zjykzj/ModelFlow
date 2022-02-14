#include <iostream>
#include "infer_engine.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

int main() {
    const char *file_name = "../assets/mnist_0.png";

    // 分别获取宽 / 高 / 通道数
    int x, y, n;
    // 返回的data是一个字节数组，包含了解析的图像像素值
    unsigned char *data = stbi_load(file_name, &x, &y, &n, 0);
    std::cout << "width: " << x << " height: " << y << " channels: " << n << std::endl;

    auto engine = InferEngine();
    const char *model_path = "../assets/mnist_cnn.mnn";
    engine.create(model_path);
    engine.printInfo();

    float mean[1] = {0.1307f,};
    float normals[1] = {0.3081f,};

    MNN::CV::ImageFormat sourceFormat = MNN::CV::GRAY;
    MNN::CV::ImageFormat destFormat = MNN::CV::GRAY;

    engine.setInputTensor(data, x, y, n, mean, normals, sourceFormat, destFormat);
    engine.run();
    engine.getOutputTensor();

    // 释放字节数组
    stbi_image_free(data);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}