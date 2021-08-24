//
// Created by zj on 2021/8/24.
//

#include "openvino_infer.h"

OpenVINOInfer::OpenVINOInfer() {
    input_tensor = cv::Mat::zeros(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC3);
    // just wrap Mat data by Blob::Ptr
    img_blob = InferenceEngine::make_shared_blob<float>(
            {InferenceEngine::Precision::FP32, {1, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH},
             InferenceEngine::Layout::NHWC},
            (float *) input_tensor.data);
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

bool OpenVINOInfer::infer(const cv::Mat &img, std::vector<float> &output_values) {
    img.copyTo(input_tensor);

    // without allocating of new memory
    // infer_request accepts input blob of any size
    infer_request.SetBlob(input_name, img_blob);
    // -----------------------------------------------------------------------------------------------------

    // Step 7. Do inference
    /* Running the request synchronously */
    infer_request.Infer();
    // -----------------------------------------------------------------------------------------------------

    // Step 8. Process output
    Blob::Ptr output = infer_request.GetBlob(output_name);
    auto const memLocker = output->cbuffer(); // use const memory locker
    // output_buffer is valid as long as the lifetime of memLocker
    const float *output_buffer = memLocker.as<const float *>();
    for (int i = 0; i < output->getTensorDesc().getDims()[1]; i++) {
//        std::cout << i << ": " << output_buffer[i] << std::endl;
        output_values.push_back(output_buffer[i]);
    }
    /** output_buffer[] - accessing output blob data **/

    return true;
}
