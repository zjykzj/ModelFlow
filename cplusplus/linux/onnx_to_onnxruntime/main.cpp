// refer to
// onnxruntime/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// ONNX-Runtime-Inference/src/inference.cpp
// https://github.com/leimao/ONNX-Runtime-Inference/blob/main/src/inference.cpp

#include <assert.h>
#include <vector>

#include <algorithm>
#include "numeric"
#include <valarray>
#include <array>
#include <cmath>

#include <onnxruntime_cxx_api.h>
//#include "image_process.h"
//#include "common_operation.h"

template<typename T>
static void softmax(T &input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

template<typename T>
T vectorProduct(const std::vector<T> &v) {
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

int main(int argc, char *argv[]) {
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // initialize session options if needed
    Ort::SessionOptions session_options;
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

    //*************************************************************************
    // create session and load model into memory
    // using squeezenet version 1.3
    // URL = https://github.com/onnx/models/tree/master/squeezenet
#ifdef _WIN32
    const wchar_t* model_path = L"/home/zj/repos/onnx/outputs/mobilenet_v1_224.onnx";
#else
    const char *model_path = "/home/zj/repos/onnx/outputs/mobilenet_v1_224.onnx";
#endif

    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path, session_options);

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char *> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
    // Otherwise need vector<vector<>>

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

    // Results should be...
    // Number of inputs = 1
    // Input 0 : name = data_0
    // Input 0 : type = 1
    // Input 0 : num_dims = 4
    // Input 0 : dim 0 = 1
    // Input 0 : dim 1 = 3
    // Input 0 : dim 2 = 224
    // Input 0 : dim 3 = 224

    //*************************************************************************
    // print model output layer (node names, types, shape etc.)

    // print number of model input nodes
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char *> output_node_names(num_output_nodes);
    std::vector<int64_t> output_node_dims;
    // simplify... this model has only 1 output node {1, N}.
    // Otherwise need vector<vector<>>

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

    //*************************************************************************
    // Score the model using sample data, and inspect values

    size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
    // use OrtGetTensorShapeElementCount() to get official size!

    std::vector<float> input_tensor_values(input_tensor_size);

    // initialize input data with values in [0.0, 1.0]
    for (unsigned int i = 0; i < input_tensor_size; i++)
//        input_tensor_values[i] = (float) i / (input_tensor_size + 1);
//        input_tensor_values[i] = 1.0 + i;
        input_tensor_values[i] = 1.0;

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                              input_tensor_size, input_node_dims.data(),
                                                              input_node_dims.size());
    assert(input_tensor.IsTensor());

    // create output tensor object
//    size_t output_tensor_size = vector_product(output_node_dims);
    size_t output_tensor_size = vectorProduct(output_node_dims);
    std::vector<float> output_tensor_values(output_tensor_size);
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_tensor_values.data(),
                                                               output_tensor_size, output_node_dims.data(),
                                                               output_node_dims.size());
    assert(output_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                &input_tensor, 1,
                output_node_names.data(), &output_tensor, 1);
    assert(output_tensor.IsTensor());

    // Get pointer to output tensor float values
    float *floatarr = output_tensor.GetTensorMutableData<float>();
//    assert(abs(floatarr[0] - 0.014135006) < 1e-6);

    // score the model, and print scores for first 5 classes
    for (int i = 0; i < output_node_dims[1]; i++)
        printf("Score for class [%d] =  %f\n", i, floatarr[i]);

    softmax(output_tensor_values);
    for (int i = 0; i < output_tensor_values.size(); i++)
        printf("Score for class [%d] =  %f\n", i, floatarr[i]);

    printf("Done!\n");
    return 0;
}