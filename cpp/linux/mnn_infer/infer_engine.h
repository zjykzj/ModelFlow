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
     * 创建解释器Interpreter
     * 创建会话Session
     * 会话配置
     * 预处理配置
     * @param model_path
     */
    void create(const char *model_path);

    /**
     * 打印模型相关信息，包括：
     * 1. 模型：内存占用、计算量FLOPs、后端引擎类型、批量大小
     * 2. 输入：图像长 / 图像宽 / 通道数 / 数据类型 / 数组长度
     * 3. 输出：
     */
    void printInfo();

    /**
     * 图像预处理
     * 输入图像数据
     *
     * @param data
     * @param data_size
     */
    void setInputTensor(const unsigned char *inputImage, int width, int height, int channels,
                        float *mean, float *std,
                        MNN::CV::ImageFormat srcFormat, MNN::CV::ImageFormat dstFormat);

    /**
     * 模型推理
     */
    void run();

    /**
     * 获取输出结果
     */
    void getOutputTensor();

private:
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session *session{};
};


#endif //MNN_INFER_INFER_ENGINE_H
