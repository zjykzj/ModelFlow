#include <iostream>
#include "chrono"

#include "Model.h"


int main() {
#ifdef _WIN32
    const wchar_t* model_path = L"/home/zj/repos/onnx/outputs/mobilenet_v1_224.onnx";
#else
    const char *model_path = "/home/zj/repos/onnx/outputs/mobilenet_v1_224.onnx";
#endif

    Model *model = new Model(model_path);

    // Measure latency
    int numTests{100};
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < numTests; i++) {
        std::string imageFilepath{"/home/zj/opencv/opencv-4.4.0/opencv/samples/data/lena.jpg"};
        model->imagePreprocess(imageFilepath);

        int64_t res = model->Run();
//        std::cout << res << std::endl;
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Minimum Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() /
                 static_cast<float>(numTests)
              << " ms" << std::endl;

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
