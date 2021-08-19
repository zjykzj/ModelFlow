//
// Created by zj on 2021/8/19.
//


#include "InferEngine.h"

InferEngine::InferEngine(const char *model_path) {
    session_options.SetIntraOpNumThreads(1);
    // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
    // session (we also need to include cuda_provider_factory.h above which defines it)
    // #include "cuda_provider_factory.h"
    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session = Ort::Session(env, model_path, session_options);

    size_t num_input_nodes = session.GetInputCount();
    input_node_names = std::vector<const char *>(num_input_nodes);

    size_t num_output_nodes = session.GetOutputCount();
    output_node_names = std::vector<const char *>(num_output_nodes);
}

void InferEngine::release() {

}

void InferEngine::print_input_info() {
    Ort::AllocatorWithDefaultOptions allocator;

    // print model input layer (node names, types, shape etc.)
    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char *input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        // Set the size of dimension 0 to 1
        input_node_dims[0] = 1;
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }
}

void InferEngine::print_output_info() {
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model output nodes
    size_t num_output_nodes = session.GetOutputCount();
    printf("Number of outputs = %zu\n", num_output_nodes);

    // iterate over all output nodes
    for (int i = 0; i < num_output_nodes; i++) {
        // print output node names
        char *output_name = session.GetOutputName(i, allocator);
        printf("Output %d : name=%s\n", i, output_name);
        output_node_names[i] = output_name;

        // print output node types
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);

        // print output shapes/dims
        output_node_dims = tensor_info.GetShape();
        // Set the size of dimension 0 to 1
        output_node_dims[0] = 1;
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < output_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }
    //*************************************************************************
    // Similar operations to get output node information.
    // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
    // OrtSessionGetOutputTypeInfo() as shown above.
}

void InferEngine::infer(const cv::Mat &img, std::vector<float> &output_values) {
    // Score the model using sample data, and inspect values
    std::vector<float> input_tensor_values(input_tensor_size);
    input_tensor_values.assign(img.begin<float>(), img.end<float>());

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              input_tensor_values.data(),
                                                              input_tensor_size,
                                                              input_node_dims.data(),
                                                              input_node_dims.size());
    assert(input_tensor.IsTensor());

    // create output tensor object
    size_t output_tensor_size = vector_product(output_node_dims);
    std::vector<float> output_tensor_values(output_tensor_size);
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                               output_tensor_values.data(),
                                                               output_tensor_size,
                                                               output_node_dims.data(),
                                                               output_node_dims.size());
    assert(output_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    session.Run(Ort::RunOptions{nullptr},
                input_node_names.data(),
                &input_tensor, 1,
                output_node_names.data(),
                &output_tensor, 1);
    assert(output_tensor.IsTensor());

    // Get pointer to output tensor float values
    auto *floatarr = output_tensor.GetTensorMutableData<float>();

//    std::vector<float> output_values;
    // score the model, and print scores for first 5 classes
    for (int i = 0; i < output_node_dims[1]; i++) {
//        printf("Score for class [%d] =  %f\n", i, floatarr[i]);
        output_values.push_back(floatarr[i]);
    }
}