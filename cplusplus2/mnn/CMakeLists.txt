cmake_minimum_required(VERSION 3.24)
project(mnn)

set(CMAKE_CXX_STANDARD 17)

# ----------------------------------------------------------------------------------- MNN
set(MNN_VERSION "2.4.0")
get_filename_component(ABSOLUTE_MNN_DIR ../../3rdparty/mnn/${MNN_VERSION} ABSOLUTE)

set(MNN_INCLUDE_DIR ${ABSOLUTE_MNN_DIR}/include/)
include_directories(${MNN_INCLUDE_DIR})

set(MNN_LIB ${ABSOLUTE_MNN_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}/libMNN.so)

MESSAGE(STATUS "INFER_INCLUDE_DIR: ${INFER_INCLUDE_DIR}")
MESSAGE(STATUS "推理库: ${INFER_ROOT}")

# ----------------------------------------------------------------------------------- stb
get_filename_component(ABSOLUTE_STB_DIR ../../3rdparty/stb ABSOLUTE)
include_directories(${ABSOLUTE_STB_DIR})
MESSAGE(STATUS "STB_DIR: ${ABSOLUTE_STB_DIR}")

# ----------------------------------------------------------------------------------- OpenCV
get_filename_component(ABSOLUTE_OpenCV_DIR ../../3rdparty/opencv/4.5.2 ABSOLUTE)
set(OpenCV_DIR ${ABSOLUTE_OpenCV_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

# ----------------------------------------------------------------------------------- Compile
add_executable(mnn main.cpp)
target_link_libraries(mnn ${OpenCV_LIBS} ${MNN_LIB})

add_executable(pictureRecognition pictureRecognition.cpp)
target_link_libraries(pictureRecognition ${MNN_LIB})