cmake_minimum_required(VERSION 3.17)
project(onnx_to_onnxruntime)

set(CMAKE_CXX_STANDARD 14)

# set opencv
# see [get_filename_component](https://cmake.org/cmake/help/latest/command/get_filename_component.html)
get_filename_component(ABSOLUTE_OpenCV_DIR ../../../3rdparty/opencv ABSOLUTE)
# Add the installation prefix of "OpenCV" to CMAKE_PREFIX_PATH or set
# "OpenCV_DIR" to a directory containing OpenCVConfig.cmake/opencv-config.cmake.
#set(OpenCV_DIR ../../3rdparty/opencv/lib/cmake/opencv4)
set(OpenCV_DIR ${ABSOLUTE_OpenCV_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

# set onnx runtime
get_filename_component(ABSOLUTE_OnnxRuntime_DIR ../../../3rdparty/onnxruntime ABSOLUTE)
include_directories(${ABSOLUTE_OnnxRuntime_DIR}/include)
link_directories(${ABSOLUTE_OnnxRuntime_DIR}/lib)

# set zcm
include_directories(../../zco/install/include)
link_directories(../../zco/install/lib)

#add_executable(onnx_to_onnx_runtime main.cpp infer_engine.cpp infer_engine.h test_infer_engine.cpp)
add_executable(onnx_to_onnxruntime onnx_infer.cpp onnx_infer.h test_onnx_infer.cpp)
target_link_libraries(onnx_to_onnxruntime ${OpenCV_LIBS})
target_link_libraries(onnx_to_onnxruntime onnxruntime)
target_link_libraries(onnx_to_onnxruntime zco)