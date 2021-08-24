//
// Created by zj on 2021/8/24.
//

#include "openvino_infer.h"
#include "../classification_results.h"
#include "../common.hpp"
#include "../ocv_common.hpp"

OpenVINOInfer::OpenVINOInfer() {

}

bool OpenVINOInfer::create(const char *model_path, const char *device_name) {
    // Step 2. Read a model in OpenVINO Intermediate Representation
    // (.xml and .bin files) or ONNX (.onnx file) format
    network = ie.ReadNetwork(model_path);
    if (network.getOutputsInfo().size() != 1)
        throw std::logic_error("Sample supports topologies with 1 output only");
    if (network.getInputsInfo().size() != 1)
        throw std::logic_error("Sample supports topologies with 1 input only");

    // Step 3. Configure input & output
    // Prepare input blobs
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    input_name = network.getInputsInfo().begin()->first;

    /* Mark input as resizable by setting of a resize algorithm.
     * In this case we will be able to set an input blob of any shape to an
     * infer request. Resize and layout conversions are executed automatically
     * during inference
     * */
    //        input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    //        input_info->getPreProcess().setColorFormat(ColorFormat::RGB);
    input_info->setLayout(Layout::NHWC);
    //        input_info->setPrecision(Precision::U8);
    // Prepare output blobs
    if (network.getOutputsInfo().empty()) {
        std::cerr << "Network outputs info is empty" << std::endl;
        return EXIT_FAILURE;
    }
    DataPtr output_info = network.getOutputsInfo().begin()->second;
    output_name = network.getOutputsInfo().begin()->first;

    output_info->setPrecision(Precision::FP32);

    // Step 4. Loading a model to the device
    executable_network = ie.LoadNetwork(network, device_name);
    // Step 5. Create an infer request
    infer_request = executable_network.CreateInferRequest();
    return true;
}

bool OpenVINOInfer::release() {
    return false;
}

void OpenVINOInfer::print_input_info() {

}

void OpenVINOInfer::print_output_info() {

}

bool OpenVINOInfer::infer(const char *img_path, const cv::Mat &img, std::vector<float> &output_values) {
    // --------------------------- Step 6. Prepare input
    // --------------------------------------------------------
    /* Read input image to a blob and set it to an infer request without resize
     * and layout conversions. */
    //        auto input = infer_request.GetBlob(input_name);
    //        std::cout << input << std::endl;
    //        size_t num_channels = input->getTensorDesc().getDims()[1];
    //        size_t h = input->getTensorDesc().getDims()[2];
    //        size_t w = input->getTensorDesc().getDims()[3];
    //        size_t image_size = h * w;

    clock_t start, inner, finish;
    double duration, duration_img, duration_model;
    start = clock();

    Blob::Ptr imgBlob = wrapMat2Blob(img);     // just wrap Mat data by Blob::Ptr
    // without allocating of new memory
    infer_request.SetBlob(input_name, imgBlob);  // infer_request accepts input blob of any size
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 7. Do inference
    // --------------------------------------------------------
    /* Running the request synchronously */
    inner = clock();
    infer_request.Infer();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 8. Process output
    // ------------------------------------------------------
    Blob::Ptr output = infer_request.GetBlob(output_name);
    //            std::cout << "output size: " << output->size() << std::endl;
    // Print classification results
    //        ClassificationResult_t classificationResult(output, {input_image_path}, 1, output->size());
    ClassificationResult classificationResult(output, {img_path}, 1, 5);
    //            classificationResult.print();

    auto _results = classificationResult.getResults();
    auto _outputs = classificationResult.getOuts();
    std::vector<float> probs(_outputs);
    softmax(probs);

    //            std::cout << _results.size() << " " << _outputs.size() << std::endl;
    for (int i = 0; i < _results.size(); i++) {
        std::cout << _results.at(i) << " " << _outputs.at(i) << " " << probs.at(i) << std::endl;

        auto pred_idx = int(_results.at(i));

    }

    // -----------------------------------------------------------------------------------------------------
    finish = clock();
    duration_img = (double) (inner - start) / CLOCKS_PER_SEC;
    duration_model = (double) (finish - inner) / CLOCKS_PER_SEC;
    duration = (double) (finish - start) / CLOCKS_PER_SEC;
    printf("image process need %f seconds\n", duration_img);
    printf("model process need %f seconds\n", duration_model);
    printf("total need %f seconds\n", duration);

    return false;
}
