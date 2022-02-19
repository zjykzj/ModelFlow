//
// Created by zj on 2022/2/13.
//

#ifndef MNN_INFER_INFER_ENGINE_H
#define MNN_INFER_INFER_ENGINE_H

#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"

class InferEngine {

public:
    InferEngine();

    /**
     * 1. 创建Interpreter
     * 2. 创建Session
     * 3. 会话配置
     */
    void create(const char *model_path);

    /**
     * 打印模型信息
     * 1. 内存占用、计算量、后端类型
     * 2. 输入大小、数据类型
     * 3. 输出大小
     */
    void printModelInfo();

    /**
     * 1. 图像预处理（图像缩放、数据归一化、数据格式转换）
     * 2. 复制数据到模型
     */
    void setInputTensor(const unsigned char *inputImage, int width, int height, int channel,
                        float *means, float *normals,
                        MNN::CV::ImageFormat srcFormat, MNN::CV::ImageFormat dstFormat);

    /**
     * 模型推理
     */
    int run();

    /**
     * 获取计算结果
     */
    std::vector<std::pair<int, float>> getOutputTensor();

    /**
     * 获取输入Tensor大小
     */
    void getInputTensorShape(int &width, int &height, int &channel);

private:
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session *session{};

    int last_width{};
    int last_height{};
    std::shared_ptr<MNN::CV::ImageProcess> pretreat = nullptr;
};


#endif //MNN_INFER_INFER_ENGINE_H
