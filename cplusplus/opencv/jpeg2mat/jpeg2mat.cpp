//
// Created by zj on 24-5-1.
//

#include "jpeg2mat.h"

int Mat2Jpeg(const cv::Mat &mat_data, std::vector<unsigned char> &jpeg_buff) {
    std::vector<int> param;
    param.push_back(cv::IMWRITE_JPEG_QUALITY);
    param.push_back(95);  // default(95) 0-100
    cv::imencode(".jpg", mat_data, jpeg_buff, param);

    return 0;
}

int Jpeg2Mat(const std::vector<unsigned char> &jpeg_buff, cv::Mat &mat_data) {
    if (jpeg_buff[0] == 0xFF && jpeg_buff[1] == 0xD8)
        mat_data = cv::imdecode(jpeg_buff, cv::ImreadModes::IMREAD_COLOR);
    else {
        std::cerr << "jpeg_buff format error" << std::endl;
        exit(1);
    }

    return 0;
}