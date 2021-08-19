//
// Created by zj on 2021/8/19.
//

#include "InferEngine.h"
#include "image_process.h"

int main(int argc, char *argv[]) {
    // init model
    const char *model_path = "../../../../assets/demo.onnx";
    auto model = InferEngine(model_path);
    model.print_input_info();
    model.print_output_info();

    // init img
    const char *img_path = "../../../../assets/demo.jpg";
    cv::Mat img;
    read_image(img_path, img, true);
    print_image_info(img);

    square_padding(img);
    print_image_info(img);

    cv::Mat resize_img;
    resize(img, resize_img, cv::Size2i(model.IMAGE_WIDTH, model.IMAGE_HEIGHT));
    print_image_info(resize_img);

    normalize(resize_img, true, cv::Scalar(0.45, 0.45, 0.45), cv::Scalar(0.225, 0.225, 0.225), 255.0);
    print_image_info(resize_img);

    cv::Mat dst;
    hwc_2_chw(resize_img, dst);

    // model infer
    std::vector<float> output_values;
    model.infer(dst, output_values);

    // get max index
    size_t max_idx = get_max_idx(output_values);
    std::cout << "max_idx: " << max_idx << std::endl;

    // sort
    std::vector<size_t> output_sorted_idxes(output_values.size());
    get_top_n(output_values, output_sorted_idxes);

    // softmax
    std::vector<float> output_probs(output_values);
    softmax(output_probs);
    for (int i = 0; i < output_probs.size(); i++) {
        auto idx = int(output_sorted_idxes[i]);
        printf("Score for index [%d] =  %f %f\n", idx, output_values.at(idx), output_probs[idx]);
    }

    printf("Done!\n");
    return 0;
}