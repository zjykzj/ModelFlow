//
// Created by zj on 24-5-1.
//

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

cv::Mat resize_and_crop(const cv::Mat &image) {
    // 获取图像的尺寸
    int h = image.rows;
    int w = image.cols;

    // 计算缩放比例
    int new_h, new_w;
    if (h < w) {
        new_h = 256;
        new_w = static_cast<int>(w * 256 / h);
    } else {
        new_h = static_cast<int>(h * 256 / w);
        new_w = 256;
    }

    // 缩放图像
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_w, new_h));

    // 计算中心裁剪区域
    int top = (new_h - 224) / 2;
    int left = (new_w - 224) / 2;
    int bottom = top + 224;
    int right = left + 224;

    // 中心裁剪图像
    cv::Rect roi(left, top, 224, 224);
    cv::Mat cropped_image = resized_image(roi);

    std::cout << "cropped_image.shape: " << cropped_image.size() << " - dtype: " << cropped_image.type() << std::endl;
    return cropped_image;
}

cv::Mat preprocess_image(const cv::Mat &image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // 缩放图像
    cv::Mat resizedImage = resize_and_crop(image);

    // 均值归一化
    cv::Scalar mean(0.485, 0.456, 0.406);            // RGB 均值
    cv::Scalar stdDev(0.229, 0.224, 0.225);          // RGB 标准差
    resizedImage.convertTo(resizedImage, CV_32FC3);  // 转换为 float 类型
    resizedImage /= 255.0;                           // 归一化到 [0, 1] 范围
    cv::subtract(resizedImage, mean, resizedImage);  // 减去均值
    cv::divide(resizedImage, stdDev, resizedImage);  // 除以标准差

    // 准备输入数据
    cv::Mat blob = cv::dnn::blobFromImage(resizedImage);

    return blob;
}

cv::Mat preprocess_image2(const cv::Mat &image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // 缩放图像
    cv::Mat resizedImage = resize_and_crop(image);

    // 均值归一化
    cv::Scalar mean(0.485, 0.456, 0.406);    // RGB 均值
    cv::Scalar stdDev(0.229, 0.224, 0.225);  // RGB 标准差

    // 准备输入数据
    cv::Mat blob = cv::dnn::blobFromImage(resizedImage, 1 / 255.0, cv::Size(224, 224), mean * 255, false);
    // Check std values.
    if (stdDev.val[0] != 0.0 && stdDev.val[1] != 0.0 && stdDev.val[2] != 0.0) {
        cv::divide(blob, stdDev, blob);  // 除以标准差
    }

    return blob;
}

int main() {
    // 加载 ONNX 模型
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../../../export/resnet50_pytorch.onnx");
    // 检查模型是否成功加载
    if (net.empty()) {
        std::cerr << "Failed to load ONNX model." << std::endl;
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 加载图像
    cv::Mat image = cv::imread("../../../assets/imagenet/n02113023/ILSVRC2012_val_00010244.JPEG");
    // 检查图像是否成功加载
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    cv::Mat blob = preprocess_image(image);
    //    cv::Mat blob = preprocess_image2(image);

    // 设置输入数据
    net.setInput(blob);
    // 执行前向传播
    cv::Mat prob = net.forward();
    // 解析输出结果
    cv::Mat probMat = prob.reshape(1, 1);  // 将结果转换为一维矩阵
    for (int i = 0; i < 10; ++i) {
        float outputValue = probMat.at<float>(0, i);
        std::cout << " " << outputValue;
    }
    std::cout << std::endl;

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
