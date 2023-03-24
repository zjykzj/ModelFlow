//
// Created by zj on 23-3-24.
//

#include "yolodetector.h"

cv::Mat YOLODetector::FormatYOLOv5(const cv::Mat &source) {
	int col = source.cols;
	int row = source.rows;
	int _max = MAX(col, row);
	cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
	source.copyTo(result(cv::Rect(0, 0, col, row)));
	return result;
}

int YOLODetector::Init(const std::string &model_path, bool is_cuda) {
	this->net_ = std::make_shared<cv::dnn::Net>(cv::dnn::readNet(model_path.c_str()));
	if (is_cuda) {
		std::cout << "Attempt to use CUDA\n";
		this->net_->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		this->net_->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	} else {
		std::cout << "Running on CPU\n";
		this->net_->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		this->net_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

	return 0;
}
int YOLODetector::Detect(const cv::Mat &image,
						 const std::vector<std::string> &class_names,
						 std::vector<BoxInfo> &output) {
	cv::Mat blob;
	auto input_image = this->FormatYOLOv5(image);
	cv::dnn::blobFromImage(input_image,
						   blob,
						   1. / 255.,
						   cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
						   cv::Scalar(),
						   true,
						   false);

	this->net_->setInput(blob);
	std::vector<cv::Mat> outputs;
	this->net_->forward(outputs, this->net_->getUnconnectedOutLayersNames());

	float x_factor = input_image.cols / INPUT_WIDTH;
	float y_factor = input_image.rows / INPUT_HEIGHT;

	float *data = (float *)outputs[0].data;

	const int dimensions = 85;
	const int rows = 25200;

	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	for (int i = 0; i < rows; ++i) {
		float confidence = data[4];
		if (confidence >= CONFIDENCE_THRESHOLD) {
			float *classes_scores = data + 5;
			cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
			cv::Point class_id;
			double max_class_score;
			minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			if (max_class_score > SCORE_THRESHOLD) {

				confidences.push_back(confidence);

				class_ids.push_back(class_id.x);

				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];
				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
		data += 85;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
	for (int idx : nms_result) {
		BoxInfo result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}

	return 0;
}
