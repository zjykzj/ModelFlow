cmake_minimum_required(VERSION 3.17)
project(onnx_to_onnxruntime)

set(CMAKE_CXX_STANDARD 14)

# set onnx runtime
set(CMAKE_PREFIX_PATH /home/zj/software/onnxruntime/onnxruntime-linux-x64-1.8.0)
include_directories(/home/zj/software/onnxruntime/onnxruntime-linux-x64-1.8.0/include)
find_library(ONNXRUNTIME onnxruntime /home/zj/software/onnxruntime/onnxruntime-linux-x64-1.8.0/lib)

MESSAGE("ONNXRUNTIME: ${ONNXRUNTIME}")

add_executable(onnx_to_onnxruntime main.cpp)
target_link_libraries(onnx_to_onnxruntime ${ONNXRUNTIME})