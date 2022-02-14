//
// Created by zj on 2022/2/13.
//

#include "infer_engine.h"

InferEngine::InferEngine() = default;

void InferEngine::create(const char *model_path) {
    net = std::shared_ptr<MNN::Interpreter>(
//            MNN::Interpreter::createFromBuffer((char *) mnist_cnn_mnn, mnist_cnn_mnn_len)
            MNN::Interpreter::createFromFile(model_path)
    );

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_AUTO;
    // BackendConfig bnconfig;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // config.backendConfig = &bnconfig;
    session = net->createSession(config);

    // 设置图像输入批量为1
    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    MNN_PRINT("b: shape[0]: %d, shape[1]: %d, shape[2]: %d, shape[3]: %d\n", shape[0], shape[1], shape[2], shape[3]);
    // Set Batch Size
    shape[0] = 1;
    MNN_PRINT("a: shape[0]: %d, shape[1]: %d, shape[2]: %d, shape[3]: %d\n", shape[0], shape[1], shape[2], shape[3]);
    net->resizeTensor(input, shape);
    net->resizeSession(session);
}

void InferEngine::printInfo() {
    // 模型
    float memoryUsage = 0.0f;
    net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryUsage);
    float flops = 0.0f;
    net->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    int backendType[2];
    net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);

    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    MNN_PRINT("Session Info: memory use %f MB, flops is %f M, backendType is %d, batch size = %d\n", memoryUsage, flops,
              backendType[0], shape[0]);

    // 输入
    std::shared_ptr<MNN::Tensor> inputUser(new MNN::Tensor(input, MNN::Tensor::CAFFE));
    MNN_PRINT("Input Info: \n");
    input->printShape();
    MNN_PRINT("size: %d elementSize: %d data type: %d\n",
              inputUser->size(), inputUser->elementSize(), inputUser->getType().code);
    auto bpp = inputUser->channel();
    auto size_h = inputUser->height();
    auto size_w = inputUser->width();
    MNN_PRINT("w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

    // 输出
    auto output = net->getSessionOutput(session, NULL);
    MNN_PRINT("Output Info: \n");
    output->printShape();
}

void InferEngine::setInputTensor(const unsigned char *inputImage, int width, int height, int channels,
                                 float *mean, float *std,
                                 MNN::CV::ImageFormat srcFormat, MNN::CV::ImageFormat dstFormat) {
    auto input = net->getSessionInput(session, NULL);
    std::shared_ptr<MNN::Tensor> inputUser(new MNN::Tensor(input, MNN::Tensor::CAFFE));
    auto shape = input->shape();
    auto size_h = shape[2];
    auto size_w = shape[3];
    auto bpp = shape[1];

    // 图像预处理
    MNN::CV::Matrix trans;
    // Set transform, from dst scale to src, the ways below are both ok
    trans.setScale((float) (width - 1) / (size_w - 1), (float) (height - 1) / (size_h - 1));

    MNN::CV::ImageProcess::Config cv_config;
    cv_config.filterType = MNN::CV::BILINEAR;
    // 归一化操作
    ::memcpy(cv_config.mean, mean, sizeof(mean));
    ::memcpy(cv_config.normal, std, sizeof(std));
    // 指定预处理前后图像格式
    cv_config.sourceFormat = srcFormat;
    cv_config.destFormat = dstFormat;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(cv_config));
    pretreat->setMatrix(trans);
    pretreat->convert((uint8_t *) inputImage, width, height, 0,
                      inputUser->host<float>(),
                      size_w, size_h, bpp, 0, inputUser->getType());

    input->copyFromHostTensor(inputUser.get());
}


void InferEngine::run() {
    net->runSession(session);
}

void InferEngine::getOutputTensor() {
    auto output = net->getSessionOutput(session, NULL);
    auto dimType = output->getDimensionType();
    std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, dimType));
    output->copyToHostTensor(outputUser.get());

    auto type = outputUser->getType();
    auto size = outputUser->stride(0);
    MNN_PRINT("type: %d size: %d\n", type.code, size);
    std::vector<std::pair<int, float>> tempValues(size);

    auto values = outputUser->host<float>();
    for (int i = 0; i < size; ++i) {
        tempValues[i] = std::make_pair(i, values[i]);
        MNN_PRINT("%d - %f\n", i, values[i]);
    }
}