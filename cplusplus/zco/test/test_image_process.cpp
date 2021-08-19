//
// Created by zj on 2021/8/19.
//

#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"
#include "../source/image_process.h"


int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "you should input IMG_PATH" << std::endl;
        exit(-1);
    }

    auto img_path = argv[1];

    cv::Mat img;
    read_image(img_path, img, true);

    std::cout << img.size << std::endl;
    cv::imshow("img", img);
    //    cv::waitKey(0);

    square_padding(img);
    std::cout << img.size << std::endl;
    cv::imshow("square_padding", img);
    //    cv::waitKey(0);

    cv::Mat dst;
    resize(img, dst, cv::Size2i(300, 400));
    cv::imshow("resize", dst);
    cv::waitKey(0);

    std::cout << dst.depth() << std::endl;
    std::cout << dst.type() << std::endl;
    normalize(dst, false);
    std::cout << dst.depth() << std::endl;
    std::cout << dst.type() << std::endl;

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
