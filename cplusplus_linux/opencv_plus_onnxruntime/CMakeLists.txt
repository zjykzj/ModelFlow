cmake_minimum_required(VERSION 3.17)
project(opencv_plus_onnxruntime)
# set opencv
set(CMAKE_PREFIX_PATH /home/zj/opencv/opencv-4.4.0/install)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/home/zj/opencv/opencv-4.4.0/opencv/modules")
include_directories("/home/zj/opencv/opencv-4.4.0/opencv_contrib/modules")

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

# set onnx runtime
set(CMAKE_PREFIX_PATH /home/zj/software/onnxruntime/onnxruntime-linux-x64-1.8.0)
include_directories(/home/zj/software/onnxruntime/onnxruntime-linux-x64-1.8.0/include)
find_library(ONNX_RUNTIME_LIBS onnxruntime /home/zj/software/onnxruntime/onnxruntime-linux-x64-1.8.0/lib)

MESSAGE("ONNX_RUNTIME_LIBS: ${ONNX_RUNTIME_LIBS}")

set(CMAKE_CXX_STANDARD 14)

add_executable(opencv_plus_onnxruntime main.cpp)
target_link_libraries(opencv_plus_onnxruntime ${ONNX_RUNTIME_LIBS})
target_link_libraries(opencv_plus_onnxruntime ${OpenCV_LIBS})