#include <fstream>
#include <opencv2/opencv.hpp>

#include "src/yolodetector.h"

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
                                        cv::Scalar(255, 0, 0)};

std::vector<std::string> load_class_names(const std::string &class_path) {
    std::vector<std::string> class_names;
    std::ifstream ifs(class_path);
    std::string line;
    while (getline(ifs, line)) {
        class_names.push_back(line);
    }
    return class_names;
}

int DrawImage(cv::Mat &image, const std::vector<BoxInfo> &output, const std::vector<std::string> &class_names) {
    for (auto detection : output) {
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto &color = colors[classId % colors.size()];
        cv::rectangle(image, box, color, 3);

        cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        std::ostringstream text_info;
        text_info << class_names[classId];
        text_info << " ";
        text_info << std::fixed << std::setprecision(3);
        text_info << detection.confidence;
        cv::putText(image, text_info.str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
    }

    return 0;
}

int main(int argc, char **argv) {
//    cv::Mat image = cv::imread("../../../assets/bus.jpg", cv::IMREAD_COLOR);
        cv::Mat image = cv::imread("../../../assets/zidane.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error opening image file\n";
        return -1;
    }
    std::vector<std::string> class_names = load_class_names("../../../assets/coco.names");
    if (class_names.empty()) {
        std::cerr << "Error opening file\n";
        return -1;
    }
    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    YOLODetector yolo_detector;
    yolo_detector.Init("../../../yolov5s.onnx", is_cuda);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<BoxInfo> output;
    yolo_detector.Detect(image, class_names, output);

    // 计算时间差
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // 将持续时间转换为毫秒并输出
    double milliseconds = duration.count() * 1000;
    std::cout << "Time taken: " << milliseconds << " milliseconds" << std::endl;

    // 打印输出的长度
    std::cout << "Output size: " << output.size() << std::endl;

    DrawImage(image, output, class_names);
    cv::imshow("output", image);
    cv::waitKey(0);

    return 0;
}