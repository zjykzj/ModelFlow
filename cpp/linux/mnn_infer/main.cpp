#include <iostream>
#include "infer_engine.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

void find_max(std::vector<std::pair<int, float>> &data) {
    // Find Max
    std::sort(data.begin(), data.end(),
              [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });
}

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
    engine.printModelInfo();

    float means[3] = {0.f, 0.f, 0.f};
    float normal = 1.0 / 256;
    float normals[3] = {normal, normal, normal};

    MNN::CV::ImageFormat sourceFormat = MNN::CV::RGB;
    MNN::CV::ImageFormat destFormat = MNN::CV::RGB;

    engine.setInputTensor(data, x, y, n, means, normals, sourceFormat, destFormat);
    engine.run();
    std::vector<std::pair<int, float>> tmpValues = engine.getOutputTensor();

    int tmp_size = tmpValues.size();
    int length = tmp_size > 10 ? 10 : tmp_size;
    for (int i = 0; i < length; i++) {
        MNN_PRINT("%d, %f\n", tmpValues[i].first, tmpValues[i].second);
    }

    MNN_PRINT("sorted...\n");
    find_max(tmpValues);
    for (int i = 0; i < length; i++) {
        MNN_PRINT("%d, %f\n", tmpValues[i].first, tmpValues[i].second);
    }

    // 释放字节数组
    stbi_image_free(data);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}