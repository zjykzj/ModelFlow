cmake_minimum_required(VERSION 3.24)
project(mnn)

set(CMAKE_CXX_STANDARD 17)

MESSAGE(STATUS "Using Linux")
set(INFER_LIB_VERSION "1.2.7")
set(EXT_LIB pthread)

set(INFER_ROOT ${CMAKE_SOURCE_DIR}/3rdparty/mnn/${INFER_LIB_VERSION})
set(INFER_INCLUDE_DIR ${INFER_ROOT}/include/)
set(INFER_LIB ${INFER_ROOT}/lib/x86_64/libMNN.a)

MESSAGE(STATUS "推理库: ${INFER_ROOT}")
include_directories(${INFER_INCLUDE_DIR})

# set opencv
# see [get_filename_component](https://cmake.org/cmake/help/latest/command/get_filename_component.html)
get_filename_component(ABSOLUTE_OpenCV_DIR 3rdparty/opencv/4.5.2 ABSOLUTE)
# Add the installation prefix of "OpenCV" to CMAKE_PREFIX_PATH or set
# "OpenCV_DIR" to a directory containing OpenCVConfig.cmake/opencv-config.cmake.
#set(OpenCV_DIR ../../3rdparty/opencv/lib/cmake/opencv4)
set(OpenCV_DIR ${ABSOLUTE_OpenCV_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

add_executable(mnn main.cpp)
target_link_libraries(mnn ${OpenCV_LIBS} ${INFER_LIB})

include_directories(${CMAKE_SOURCE_DIR}/assets/mnn/)
add_executable(ClassifyInfer classifyinfer.cpp classifyinfer_test.cpp)
target_link_libraries(ClassifyInfer ${OpenCV_LIBS} ${INFER_LIB})