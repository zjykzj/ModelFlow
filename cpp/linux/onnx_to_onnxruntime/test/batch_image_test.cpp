//
// Created by zj on 2021/8/23.
//

#include "../source/infer_engine.h"
#include "image_process.h"
#include "common_operation.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image_file> <path_to_label_file>" << std::endl;
        return EXIT_FAILURE;
    }

    const char *model_path = argv[1];
    const char *img_file_path = argv[2];
    const char *label_file_path = argv[3];
    printf("model path is %s\nimg file path is %s\nimg file path is %s\n", model_path, img_file_path, label_file_path);

    auto model = new InferEngine();
    model->create(model_path);

    // 获取图片列表
    std::vector<std::string> img_list;
    read_txt_file(img_file_path, &img_list);

    // 获取标签列表
    std::vector<std::string> label_list;
    read_txt_file(label_file_path, &label_list);

    if (img_list.size() != label_list.size()) {
        std::cout << "图片列表和标签列表长度不匹配" << std::endl;
    }

    int current_top1_num = 0;
    int current_top5_num = 0;
    float total_img_read = 0;
    float total_img_preprocess = 0;
    float total_model_infer = 0;
    float total_model_postprocess = 0;
    float total_time = 0;

    for (int img_idx = 0; img_idx < img_list.size(); img_idx++) {
        auto img_path = img_list.at(img_idx);
        auto truth_label{std::stoi(label_list.at(img_idx))};
        std::cout << "img_idx: " << img_idx << " img_path: " << img_path << " " << "truth_label: " << truth_label
                  << std::endl;

        clock_t start, end, clock_img_read, clock_img_preprocess, clock_model_infer;
        double duration, duration_img_read, duration_img_preprocess, duration_model_infer, duration_model_postprocess;

        start = clock();
        cv::Mat img;
        read_image(img_path.c_str(), img, true);
        print_image_info(img);
        clock_img_read = clock();

        cv::Mat preprocess_img;
        InferEngine::preprocess(img, preprocess_img);
        clock_img_preprocess = clock();

        std::vector<float> output_values;
        model->infer(preprocess_img, output_values);
        clock_model_infer = clock();

        std::vector<size_t> output_idxes;
        std::vector<float> output_probes(output_values.size());
        InferEngine::postprocess(output_values, output_idxes, output_probes);

        for (int i = 0; i < 5; i++) {
            size_t index = output_idxes.at(i);
            printf("output idx: %zu, output value: %f, output probes: %f\n",
                   index, output_values.at(index), output_probes.at(index));

            if (index == truth_label) {
                if (i == 0) {
                    current_top1_num += 1;
                }
                current_top5_num += 1;
            }
        }
        model->release();
        end = clock();

        duration = (float) (end - start) / CLOCKS_PER_SEC;
        duration_img_read = (float) (clock_img_read - start) / CLOCKS_PER_SEC;
        duration_img_preprocess = (float) (clock_img_preprocess - clock_img_read) / CLOCKS_PER_SEC;
        duration_model_infer = (float) (clock_model_infer - clock_img_preprocess) / CLOCKS_PER_SEC;
        duration_model_postprocess = (float) (end - clock_model_infer) / CLOCKS_PER_SEC;

        std::cout << "image read: " << duration_img_read << std::endl;
        std::cout << "image process: " << duration_img_preprocess << std::endl;
        std::cout << "model infer: " << duration_model_infer << std::endl;
        std::cout << "post process: " << duration_model_postprocess << std::endl;
        std::cout << "total infer: " << duration << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;

        total_img_read += duration_img_read;
        total_img_preprocess += duration_img_preprocess;
        total_model_infer += duration_model_infer;
        total_model_postprocess += duration_model_postprocess;
        total_time += duration;
    }
    int total_iterations = int(img_list.size());

    std::cout << "total iteration: " << total_iterations << std::endl;
    std::cout << "average one img read need: " << total_img_read * 1.0 / total_iterations << std::endl;
    std::cout << "average one img preprocess need: " << total_img_preprocess * 1.0 / total_iterations << std::endl;
    std::cout << "average one model infer need: " << total_model_infer * 1.0 / total_iterations << std::endl;
    std::cout << "average one model postprocess need: " << total_model_postprocess * 1.0 / total_iterations
              << std::endl;
    std::cout << "average one iteration need: " << total_time * 1.0 / total_iterations << std::endl;
    std::cout << "top1 num: " << current_top1_num << " acc: " << float(current_top1_num * 1.0 / total_iterations)
              << std::endl;
    std::cout << "top5 num: " << current_top5_num << " acc: " << float(current_top5_num * 1.0 / total_iterations)
              << std::endl;

    return 0;
}