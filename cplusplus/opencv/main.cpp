#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat srcImg = cv::imread("../../../assets/bus.jpg");
    if (srcImg.empty()) {
        std::cerr << "LOAD IMAGE ERROR" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::imshow("src", srcImg);
    cv::waitKey(0);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
