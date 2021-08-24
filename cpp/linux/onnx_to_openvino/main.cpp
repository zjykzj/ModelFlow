
#include <inference_engine.hpp>
#include <iterator>
#include <string>

#include "image_process.h"
#include "source/infer_engine.h"

using namespace InferenceEngine;


int main(int argc, char *argv[]) {
    // ------------------------------ Parsing and validation of input arguments
    // ---------------------------------
    if (argc != 4) {
        std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string input_model{argv[1]};
    const std::string input_image_path{argv[2]};
    const std::string device_name{argv[3]};

    clock_t start, end, clock_img_read, clock_img_preprocess, clock_model_create, clock_model_infer;

    start = clock();
    cv::Mat img;
    read_image(input_image_path.c_str(), img, true);
    print_image_info(img);
    clock_img_read = clock();

    cv::Mat preprocess_img;
    InferEngine::preprocess(img, preprocess_img);
    clock_img_preprocess = clock();

    auto model = new InferEngine();
    model->create(input_model.c_str(), device_name.c_str());
    clock_model_create = clock();

    std::vector<float> output_values;
    model->infer(input_image_path.c_str(), preprocess_img, output_values);
    model->release();
    clock_model_infer = clock();

    end = clock();
    std::cout << "image read: " << (double) (clock_img_read - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "image preprocess: " << (double) (clock_img_preprocess - clock_img_read) / CLOCKS_PER_SEC << std::endl;
    std::cout << "model create: " << (double) (clock_model_create - clock_img_preprocess) / CLOCKS_PER_SEC << std::endl;
    std::cout << "model infer: " << (double) (clock_model_infer - clock_model_create) / CLOCKS_PER_SEC << std::endl;
    std::cout << "post process: " << (double) (end - clock_model_infer) / CLOCKS_PER_SEC << std::endl;
    std::cout << "total infer: " << (double) (end - start) / CLOCKS_PER_SEC << std::endl;

    return EXIT_SUCCESS;
}
