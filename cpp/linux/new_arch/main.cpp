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
    std::vector<int> output_idxes;
    engine.infer(dst, output_values, output_idxes);

    std::vector<float> probes;
    std::copy(output_values.begin(), output_values.end(), probes.begin());
    InferEngine::probes(probes);

    engine.release();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
