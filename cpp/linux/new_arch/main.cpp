#include <iostream>

#include "infer_engine.h"
#include "image_process.h"

int main() {
    auto engine = InferEngine();

    const char *img_path = "";
    cv::Mat src;
    read_image(img_path, src, true);

    const char *model_path = "";
    engine.create(model_path);
    cv::Mat dst;
    InferEngine::preprocess(src, dst);

    std::vector<float> output_values;
    engine.infer(dst, output_values);
    engine.release();

    std::vector<size_t> output_idxes;
    std::vector<float> probes(output_values.size());
    InferEngine::postprocess(output_values, output_idxes, probes);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
