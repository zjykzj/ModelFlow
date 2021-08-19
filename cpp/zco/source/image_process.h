//
// Created by zj on 2021/8/17.
//

#ifndef ZCM_IMAGE_PROCESS_H
#define ZCM_IMAGE_PROCESS_H

#include "opencv2/opencv.hpp"


/**
 * 读取图像，并转换成指定格式（默认是BGR）
 * @param img_path
 * @param dst
 * @param is_rgb
 * @return
 */
bool read_image(const char *img_path, cv::Mat &dst, bool is_rgb = false);

/**
 * 将图像填充成宽高一致
 * @param img
 * @return
 */
bool square_padding(cv::Mat &img);

/**
 * 图像缩放
 * @param orig_img
 * @param dst
 * @param size
 * @return
 */
bool resize(cv::Mat &orig_img, cv::Mat &dst, const cv::Size2i &size);

/**
 * 对输入数据执行标准化操作。执行如下操作：`img = (img / scale - mean) / std`
 * @param img
 * @param keep_32f：是否保持数据类型为CV_32F，默认为true；如果设置为false，那么转换图像为输入数据类型
 * @param mean
 * @param std
 * @param scale
 * @return
 */
bool normalize(cv::Mat &img, bool keep_32f = true, cv::Scalar mean = cv::Scalar(0.45, 0.45, 0.45),
               cv::Scalar std = cv::Scalar(0.225, 0.225, 0.225), float scale = 1.0);

/**
 * 转换HWC格式保存图像为CHW格式
 * @param src
 * @param dst
 * @return
 */
bool hwc_2_chw(cv::Mat &src, cv::Mat &dst);

/**
 * show cv::Mat info
 * @param img
 * @return
 */
bool print_image_info(const cv::Mat &img);

#endif //ZCM_IMAGE_PROCESS_H
