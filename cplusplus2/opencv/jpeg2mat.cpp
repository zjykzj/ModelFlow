//
// Created by zj on 23-3-17.
//

#include <fstream>
#include <opencv2/opencv.hpp>

int Mat2Jpeg(const cv::Mat &mat_data, std::vector<unsigned char> &jpeg_buff) {
	std::vector<int> param;
	param.push_back(cv::IMWRITE_JPEG_QUALITY);
	param.push_back(95); // default(95) 0-100
	cv::imencode(".jpg", mat_data, jpeg_buff, param);

	return 0;
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
	jpeg_file.write(reinterpret_cast<const char *>(jpeg_buff.data()),
					jpeg_buff.size());
	jpeg_file.flush();
	jpeg_file.close();
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

void demo2() {
	std::ifstream jpeg_file;
	jpeg_file.open("../../../assets/bus.jpg", std::ios::in | std::ios::binary);

	int file_len = 0;
	jpeg_file.seekg(0, std::ios::end);
	file_len = jpeg_file.tellg();
	jpeg_file.seekg(0, std::ios::beg);

	std::vector<unsigned char> jpeg_buff(file_len);
	jpeg_file.read(reinterpret_cast<char *>(jpeg_buff.data()), file_len);

	cv::Mat img;
	Jpeg2Mat(jpeg_buff, img);

	cv::imshow("img", img);
	cv::waitKey(0);

	cv::imwrite("jpeg2mat.jpg", img);
}

int main(int argc, char *argv[]) {
	//  demo1();
	demo2();
	return 0;
}