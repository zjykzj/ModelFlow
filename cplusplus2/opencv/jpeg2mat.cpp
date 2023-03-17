//
// Created by zj on 23-3-17.
//

#include <fstream>
#include <opencv2/opencv.hpp>

int Mat2Jpeg(const cv::Mat mat, std::vector<unsigned char> &buff) {
	std::vector<int> param;
	param.push_back(cv::IMWRITE_JPEG_QUALITY);
	param.push_back(95); // default(95) 0-100
	cv::imencode(".jpg", mat, buff, param);
}

void demo1() {
	cv::Mat src_img = cv::imread("../../../assets/bus.jpg");
	if (src_img.empty()) {
		std::cerr << "read image error" << std::endl;
		exit(1);
	}

	std::vector<unsigned char> jpeg_buff;
	Mat2Jpeg(src_img, jpeg_buff);

	std::ofstream jpeg_file;
	jpeg_file.open("mat2jpeg.jpg", std::ios::out | std::ios::binary);
	jpeg_file.write(reinterpret_cast<const char *>(jpeg_buff.data()), jpeg_buff.size());
	jpeg_file.flush();
	jpeg_file.close();
}

int Jpeg2Mat(cv::Mat &matage, std::vector<unsigned char> buff) {

}

void demo2() {
	std::ifstream jpeg_file;
	jpeg_file.open("../../../assets/bus.jpg", std::ios::in | std::ios::binary);

	unsigned char c;
	std::vector<unsigned char> jpeg_buff;
	while (!jpeg_file.eof()) {
		jpeg_file >> c;
		jpeg_buff.push_back(c);
	}

}

int main(int argc, char *argv[]) {

	return 0;
}