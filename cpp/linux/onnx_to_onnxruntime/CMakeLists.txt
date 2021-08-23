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

add_executable(single_image_test
        source/onnx_infer.cpp
        source/onnx_infer.h
        source/infer_engine.cpp
        source/infer_engine.h
        test/single_image_test.cpp)
target_link_libraries(single_image_test ${OpenCV_LIBS})
target_link_libraries(single_image_test onnxruntime)
target_link_libraries(single_image_test zco)

add_executable(batch_image_test
        source/onnx_infer.cpp
        source/onnx_infer.h
        source/infer_engine.cpp
        source/infer_engine.h
        test/batch_image_test.cpp)
target_link_libraries(batch_image_test ${OpenCV_LIBS})
target_link_libraries(batch_image_test onnxruntime)
target_link_libraries(batch_image_test zco)