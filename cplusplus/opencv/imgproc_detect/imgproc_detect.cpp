//
// Created by zj on 23-3-19.
//

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

cv::Rect DetectImg(cv::Mat &bgr) {
    int height = bgr.rows;
    int width = bgr.cols;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blur;
    cv::medianBlur(gray, blur, 3);

    cv::Mat thresh;
    cv::Canny(blur, thresh, 50, 150);

    cv::Mat opening;
    cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(10, 10));
    cv::morphologyEx(thresh, opening, cv::MorphTypes::MORPH_CLOSE, kernel, cv::Point(-1, -1), 6);
    cv::morphologyEx(opening, opening, cv::MorphTypes::MORPH_OPEN, kernel, cv::Point(-1, -1), 3);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(opening, contours, hierarchy, cv::RetrievalModes::RETR_EXTERNAL,
                     cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rect_list;
    for (int i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours.at(i));

        int area = rect.area();
        if (area < 70000) {
            continue;
        }
        if (rect.width > int(width * 0.95) or rect.height > int(height * 0.95)) {
            continue;
        }

        rect_list.push_back(rect);
    }

    cv::Rect dst_rect;
    if (rect_list.size() == 0) {
        dst_rect = cv::Rect(0, 0, width, height);
    } else {
        if (rect_list.size() == 1) {
            dst_rect = rect_list[0];
        } else {
            int min_x1 = 100000;
            int min_y1 = 100000;
            int max_x2 = -1;
            int max_y2 = -1;
            for (int i = 0; i < rect_list.size(); i++) {
                auto tmp_rect = rect_list.at(i);
                auto x1 = tmp_rect.x;
                auto y1 = tmp_rect.y;
                auto x2 = tmp_rect.x + tmp_rect.width;
                auto y2 = tmp_rect.y + tmp_rect.height;

                if (min_x1 > x1) min_x1 = x1;
                if (min_y1 > y1) min_y1 = y1;

                if (max_x2 < x2) max_x2 = x2;
                if (max_y2 < y2) max_y2 = y2;
            }

            dst_rect = cv::Rect(min_x1, min_y1, max_x2 - min_x1, max_y2 - min_y1);
        }
    }

    return dst_rect;
}

int main(int argc, char *argv[]) {
    std::string img_path = "../imgproc_detect/demo.jpg";
    cv::Mat bgr = cv::imread(img_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Imread ERROR" << std::endl;
        exit(1);
    }

    cv::Rect rect = DetectImg(bgr);
    cv::Mat dst = bgr(rect);

    cv::imshow("src", bgr);
    cv::imshow("dst", dst);
    cv::waitKey(0);

    return 0;
}