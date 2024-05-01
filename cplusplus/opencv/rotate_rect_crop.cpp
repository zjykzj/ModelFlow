//
// Created by zj on 23-3-17.
//

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    cv::Mat src_img = cv::imread("../../../assets/bus.jpg");
    if (src_img.empty()) {
        std::cerr << "read image error" << std::endl;
        exit(1);
    }

    cv::Mat mask(src_img.rows, src_img.cols, src_img.type(), cv::Scalar(0, 0, 0));
    //	cv::Point points[4] = {cv::Point(43, 394), cv::Point(188, 388), cv::Point(264, 911), cv::Point(35, 906)};
    std::vector<cv::Point> points2 = {cv::Point(43, 394), cv::Point(188, 388), cv::Point(264, 911), cv::Point(35, 906)};
    cv::fillConvexPoly(mask, points2.data(), 4, cv::Scalar(255, 255, 255));

    cv::Mat dst_img;
    src_img.copyTo(dst_img, mask = mask);

    cv::Rect rect = cv::boundingRect(points2);
    std::cout << rect << std::endl;

    cv::Mat crop_img = dst_img(rect);

    cv::imshow("mask", mask);
    cv::imshow("src_img", src_img);
    cv::imshow("dst_img", dst_img);
    cv::imshow("crop_img", crop_img);
    cv::waitKey(0);

    return 0;
}