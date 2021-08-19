cmake_minimum_required(VERSION 3.17)
project(opencv_plus_onnx_runtime)

set(CMAKE_CXX_STANDARD 14)

# set opencv
# Add the installation prefix of "OpenCV" to CMAKE_PREFIX_PATH or set
# "OpenCV_DIR" to a directory containing OpenCVConfig.cmake/opencv-config.cmake.
set(OpenCV_DIR /home/zj/software/opencv/install_4_5_2/lib/cmake/opencv4)
set(CMAKE_PREFIX_PATH /home/zj/software/opencv/install_4_5_2)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

# set onnx runtime
include_directories(/home/zj/software/onnxruntime/onnxruntime-linux-x64-1.6.0/include)
link_directories(/home/zj/software/onnxruntime/onnxruntime-linux-x64-1.6.0/lib)

# set zcm
include_directories(../../zcm/install/include)
link_directories(../../zcm/install/lib)

add_executable(opencv_plus_onnx_runtime main.cpp)
target_link_libraries(opencv_plus_onnx_runtime ${OpenCV_LIBS})
target_link_libraries(opencv_plus_onnx_runtime onnxruntime)