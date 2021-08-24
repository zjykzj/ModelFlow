//
// Created by zj on 2021/8/23.
//

#include "../source/infer_engine.h"
#include "image_process.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image>" << std::endl;
        return EXIT_FAILURE;
    }

    const char *model_path = argv[1];
    const char *img_path = argv[2];
    printf("model path is %s\nimg path is %s\n", model_path, img_path);

    clock_t start, end, clock_img_read, clock_img_preprocess, clock_model_create, clock_model_infer;

    start = clock();
    cv::Mat img;
    read_image(img_path, img, true);
    print_image_info(img);
    clock_img_read = clock();

    cv::Mat preprocess_img;
    InferEngine::preprocess(img, preprocess_img);
    clock_img_preprocess = clock();

    auto model = new InferEngine();
    model->create(model_path);
    clock_model_create = clock();

    std::vector<float> output_values;
    model->infer(preprocess_img, output_values);
    model->release();
    clock_model_infer = clock();

    std::vector<size_t> output_idxes;
    std::vector<float> output_probes(output_values.size());;
    InferEngine::postprocess(output_values, output_idxes, output_probes);

    for (int i = 0; i < 5; i++) {
        size_t index = output_idxes.at(i);
        printf("output idx: %zu, output value: %f, output probes: %f\n",
               index, output_values.at(index), output_probes.at(index));
    }

    end = clock();
    std::cout << "image read: " << (double) (clock_img_read - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << "image preprocess: " << (double) (clock_img_preprocess - clock_img_read) / CLOCKS_PER_SEC << std::endl;
    std::cout << "model create: " << (double) (clock_model_create - clock_img_preprocess) / CLOCKS_PER_SEC << std::endl;
    std::cout << "model infer: " << (double) (clock_model_infer - clock_model_create) / CLOCKS_PER_SEC << std::endl;
    std::cout << "post process: " << (double) (end - clock_model_infer) / CLOCKS_PER_SEC << std::endl;
    std::cout << "total infer: " << (double) (end - start) / CLOCKS_PER_SEC << std::endl;

    return 0;
}