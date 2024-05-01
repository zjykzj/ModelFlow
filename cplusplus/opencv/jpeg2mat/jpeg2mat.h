//
// Created by zj on 24-5-1.
//

#ifndef OPENCV_JPEG2MAT_JPET2MAT_H_
#define OPENCV_JPEG2MAT_JPET2MAT_H_

#include <iostream>
#include <opencv2/opencv.hpp>

int Mat2Jpeg(const cv::Mat &mat_data, std::vector<unsigned char> &jpeg_buff);

int Jpeg2Mat(const std::vector<unsigned char> &jpeg_buff, cv::Mat &mat_data);

#endif  // OPENCV_JPEG2MAT_JPET2MAT_H_
