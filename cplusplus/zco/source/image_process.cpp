//
// Created by zj on 2021/8/17.
//

#include "image_process.h"


bool read_image(const char *img_path, cv::Mat &dst, bool is_rgb) {
    cv::Mat src = cv::imread(img_path, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Image file " << img_path << " not found\n";
        return false;
    }

    if (is_rgb) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    } else {
        src.copyTo(dst);
    }
    return true;
}

bool square_padding(cv::Mat &img) {
    if (img.empty()) {
        std::cerr << "img is empty\n";
        return false;
    }
    int height = img.rows;
    int width = img.cols;

    int top, bottom, left, right;
    if (height > width) {
        top = 0;
        bottom = 0;
        left = (height - bottom) / 2;
        right = height - bottom - left;
    } else {
        top = (width - height) / 2;
        bottom = width - height - top;
        left = 0;
        right = 0;
    }

    int borderType = cv::BORDER_CONSTANT;
    cv::Scalar value(0, 0, 0);
    cv::copyMakeBorder(img, img, top, bottom, left, right, borderType, value);

    return true;
}

bool resize(cv::Mat &orig_img, cv::Mat &dst, const cv::Size2i &size) {
    const int MODEL_IN_WIDTH = size.width;
    const int MODEL_IN_HEIGHT = size.height;

    dst = orig_img.clone();
    if (orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
        printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
        cv::resize(orig_img, dst, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
    } else {
        std::cout << "resize: " << orig_img.size << std::endl;
    }
    return true;
}

bool normalize(cv::Mat &img, bool keep_32f, cv::Scalar mean, cv::Scalar std, float scale) {
    if (img.empty()) {
        std::cerr << "img is empty\n";
        return false;
    }

    const int depth = img.depth();
    img.convertTo(img, CV_32F, 1.0 / scale);

    cv::Mat channels[3];
    cv::split(img, channels);
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - mean[0]) / std[0];
    channels[1] = (channels[1] - mean[1]) / std[1];
    channels[2] = (channels[2] - mean[2]) / std[2];
    cv::merge(channels, 3, img);

    if (!keep_32f) {
        img.convertTo(img, depth);
    }

    return true;
}

bool hwc_2_chw(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::cerr << "src is empty\n";
        return false;
    }
    // HWC to CHW
    cv::dnn::blobFromImage(src, dst);
    return false;
}

bool print_image_info(const cv::Mat &img) {
    auto width = img.cols;
    auto height = img.rows;
    auto channel = img.channels();
    auto data_type = img.type();

    printf("width: %d - height: %d - channel: %d - data_type: %d\n",
           width, height, channel, data_type);

    return true;
}
