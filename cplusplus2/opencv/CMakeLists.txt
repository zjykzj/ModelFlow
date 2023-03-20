cmake_minimum_required(VERSION 3.24)
project(opencv)

set(CMAKE_CXX_STANDARD 17)

# set opencv
# see [get_filename_component](https://cmake.org/cmake/help/latest/command/get_filename_component.html)
get_filename_component(ABSOLUTE_OpenCV_DIR ../3rdparty/opencv/4.5.2 ABSOLUTE)
# Add the installation prefix of "OpenCV" to CMAKE_PREFIX_PATH or set
# "OpenCV_DIR" to a directory containing OpenCVConfig.cmake/opencv-config.cmake.
#set(OpenCV_DIR ../../3rdparty/opencv/lib/cmake/opencv4)
set(OpenCV_DIR ${ABSOLUTE_OpenCV_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

add_executable(opencv main.cpp)
target_link_libraries(opencv ${OpenCV_LIBS})

add_executable(RotateRectCrop rotate_rect_crop.cpp)
target_link_libraries(RotateRectCrop ${OpenCV_LIBS})

add_executable(Jpeg2Mat jpeg2mat.cpp)
target_link_libraries(Jpeg2Mat ${OpenCV_LIBS})

add_executable(ObjectDetect object_detect.cpp)
target_link_libraries(ObjectDetect ${OpenCV_LIBS})