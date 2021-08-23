//
// Created by zj on 2021/8/23.
//

#include "../source/infer_engine.h"
#include "image_process.h"

int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cerr << "Please input model path and image path\n" << std::endl;
        exit(-1);
    }

    const char *model_path = argv[1];
    const char *img_path = argv[2];
    printf("model path is %s, img path is %s\n", model_path, img_path);

    cv::Mat img;
    read_image(img_path, img, true);
    print_image_info(img);

    auto model = new InferEngine();
    model->create(model_path);

    cv::Mat preprocess_img;
    InferEngine::preprocess(img, preprocess_img);

    std::vector<float> output_values;
    std::vector<size_t> output_idxes;
    model->infer(preprocess_img, output_values, output_idxes);
    model->release();

    std::vector<float> output_probes(output_values);
    InferEngine::probes(output_probes);

    for (int i = 0; i < 5; i++) {
        size_t index = output_idxes.at(i);
        printf("output idx: %zu, output value: %f, output probes: %f\n",
               index, output_values.at(index), output_probes.at(index));
    }
}