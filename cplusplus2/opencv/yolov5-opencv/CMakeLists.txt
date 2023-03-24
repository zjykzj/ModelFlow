cmake_minimum_required(VERSION 3.24)
project(yolov5-opencv)

set(CMAKE_CXX_STANDARD 17)

# set opencv
get_filename_component(ABSOLUTE_OpenCV_DIR ../../../3rdparty/opencv/4.7.0 ABSOLUTE)
set(OpenCV_DIR ${ABSOLUTE_OpenCV_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

add_executable(yolo-video yolo-video.cpp src/yolodetector.cpp)
target_link_libraries(yolo-video ${OpenCV_LIBS})