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
     * @param model_path
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
     * 创建预处理器
     * 1. 数据归一化
     * 2. 预处理前后图像格式
     * @param channel
     * @param means
     * @param normals
     * @param srcFormat
     * @param dstFormat
     */
    void setPretreat(int channel,
                     float *means, float *normals,
                     MNN::CV::ImageFormat srcFormat, MNN::CV::ImageFormat dstFormat);


    /**
     * 1. 图像缩放
     * 2. 图像预处理
     * 3. 复制数据到模型
     * @param inputImage
     * @param width
     * @param height
     */
    void setInputTensor(const unsigned char *inputImage, int width, int height);

    /**
     * 模型推理
     * @return
     */
    int run();

    /**
     * 获取计算结果
     * @return
     */
    std::vector<std::pair<int, float>> getOutputTensor();

    /**
     * 获取输入Tensor大小
     * @param width
     * @param height
     * @param channel
     */
    void getInputTensorShape(int &width, int &height, int &channel);

private:
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session *session{};
    std::shared_ptr<MNN::CV::ImageProcess> pretreat;
};


#endif //MNN_INFER_INFER_ENGINE_H
