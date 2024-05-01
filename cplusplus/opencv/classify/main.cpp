//
// Created by zj on 24-5-1.
//

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // 加载 ONNX 模型
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../resnet18_pytorch.onnx");
    // 检查模型是否成功加载
    if (net.empty()) {
        std::cerr << "Failed to load ONNX model." << std::endl;
        return -1;
    }

    // 加载图像
    cv::Mat image = cv::imread("../../../assets/imagenet/ILSVRC2012_val_00010244.JPEG");
    // 检查图像是否成功加载
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    // 缩放图像
    cv::Size inputSize(224, 224);  // 模型所需的输入尺寸
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, inputSize);

    // 均值归一化
    cv::Scalar mean(0.485, 0.456, 0.406);            // RGB 均值
    cv::Scalar stdDev(0.229, 0.224, 0.225);          // RGB 标准差
    resizedImage.convertTo(resizedImage, CV_32FC3);  // 转换为 float 类型
    resizedImage /= 255.0;                           // 归一化到 [0, 1] 范围
    cv::subtract(resizedImage, mean, resizedImage);  // 减去均值
    cv::divide(resizedImage, stdDev, resizedImage);  // 除以标准差

    // 准备输入数据
    cv::Mat blob = cv::dnn::blobFromImage(resizedImage);

    // 设置输入数据
    net.setInput(blob);
    // 执行前向传播
    cv::Mat prob = net.forward();
    // 解析输出结果
    cv::Mat probMat = prob.reshape(1, 1);  // 将结果转换为一维矩阵

    // 计算分类概率
    cv::Mat softmaxProb;
    exp(probMat, softmaxProb);           // 计算指数
    softmaxProb /= sum(softmaxProb)[0];  // 归一化为概率分布

    // 获取前五个最可能的类别
    std::vector<int> top5Indices;
    cv::sortIdx(softmaxProb, top5Indices, cv::SORT_DESCENDING + cv::SORT_EVERY_ROW);

    // 输出前五个结果
    const int kTopN = 5;
    std::cout << "Top " << kTopN << " predictions:" << std::endl;
    for (int i = 0; i < kTopN; ++i) {
        int classIdx = top5Indices.at(i);
        float probability = softmaxProb.at<float>(classIdx);
        float outputValue = probMat.at<float>(classIdx);
        std::cout << "Class index: " << classIdx << ", Probability: " << probability
                  << ", Output value: " << outputValue << std::endl;
    }

    return 0;
}
