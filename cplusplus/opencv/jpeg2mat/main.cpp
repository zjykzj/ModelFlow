//
// Created by zj on 24-5-1.
//

#include <fstream>
#include <opencv2/opencv.hpp>

#include "jpeg2mat.h"

void demo_mat2jpeg() {
    cv::Mat src_img = cv::imread("../../../assets/bus.jpg");
    if (src_img.empty()) {
        std::cerr << "read image error" << std::endl;
        exit(1);
    }

    std::vector<unsigned char> jpeg_buff;
    Mat2Jpeg(src_img, jpeg_buff);

    std::ofstream jpeg_file;
    jpeg_file.open("mat2jpeg.jpg", std::ios::out | std::ios::binary);
    jpeg_file.write(reinterpret_cast<const char *>(jpeg_buff.data()),
                    jpeg_buff.size());
    jpeg_file.flush();
    jpeg_file.close();
}

void demo_jpeg2mat() {
    // 读取JPEG文件
    std::ifstream jpeg_file;
    jpeg_file.open("../../../assets/bus.jpg", std::ios::in | std::ios::binary);

    int file_len = 0;
    jpeg_file.seekg(0, std::ios::end);
    file_len = jpeg_file.tellg();
    jpeg_file.seekg(0, std::ios::beg);

    // 分配内存并读取文件数据
    std::vector<unsigned char> jpeg_buff(file_len);
    if (!jpeg_file.read(reinterpret_cast<char *>(jpeg_buff.data()), file_len)) {
        std::cerr << "Error: Unable to read file data" << std::endl;
        return;
    }
    jpeg_file.close();

    // 转换成cv::Mat
    cv::Mat img;
    Jpeg2Mat(jpeg_buff, img);

    cv::imshow("img", img);
    cv::waitKey(0);
    cv::imwrite("jpeg2mat.jpg", img);
}

int main(int argc, char *argv[]) {
    demo_mat2jpeg();
    demo_jpeg2mat();
    return 0;
}