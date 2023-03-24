#include <fstream>

#include <opencv2/opencv.hpp>

#include "src/yolodetector.h"

const std::vector<cv::Scalar>
	colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

std::vector<std::string> load_class_names(const std::string &class_path) {
	std::vector<std::string> class_names;
	std::ifstream ifs(class_path);
	std::string line;
	while (getline(ifs, line)) {
		class_names.push_back(line);
	}
	return class_names;
}

int DrawImage(cv::Mat &image, const std::vector<BoxInfo>& output, const std::vector<std::string> &class_names) {
	for (auto detection : output) {
		auto box = detection.box;
		auto classId = detection.class_id;
		const auto& color = colors[classId % colors.size()];
		cv::rectangle(image, box, color, 3);

		cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
		cv::putText(image,
					class_names[classId],
					cv::Point(box.x, box.y - 5),
					cv::FONT_HERSHEY_SIMPLEX,
					0.5,
					cv::Scalar(0, 0, 0));
	}

	return 0;
}

int main(int argc, char **argv) {
	cv::Mat frame;
	cv::VideoCapture capture("../../../../assets/sample.mp4");
	if (!capture.isOpened()) {
		std::cerr << "Error opening video file\n";
		return -1;
	}
	std::vector<std::string> class_list = load_class_names("../../../../assets/coco.names");

	bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
	YOLODetector yolo_detector;
	yolo_detector.Init("../../../../assets/yolov5n.onnx", is_cuda);

	auto start = std::chrono::high_resolution_clock::now();
	int frame_count = 0;
	float fps = -1;
	int total_frames = 0;

	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			std::cout << "End of stream\n";
			break;
		}

		std::vector<BoxInfo> output;
		yolo_detector.Detect(frame, class_list, output);

		DrawImage(frame, output, class_list);

		frame_count++;
		total_frames++;
		if (frame_count >= 30) {
			auto end = std::chrono::high_resolution_clock::now();
			fps = frame_count * 1000.0
				/ (float)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			frame_count = 0;
			start = std::chrono::high_resolution_clock::now();
		}

		if (fps > 0) {
			std::ostringstream fps_label;
			fps_label << std::fixed << std::setprecision(2);
			fps_label << "FPS: " << fps;
			std::string fps_label_str = fps_label.str();

			cv::putText(frame,
						fps_label_str,
						cv::Point(10, 25),
						cv::FONT_HERSHEY_SIMPLEX,
						1,
						cv::Scalar(0, 0, 255),
						2);
		}
		cv::imshow("output", frame);

		if (cv::waitKey(1) != -1) {
			capture.release();
			std::cout << "finished by user\n";
			break;
		}
	}

	std::cout << "Total frames: " << total_frames << "\n";
	return 0;
}