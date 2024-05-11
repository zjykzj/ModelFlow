cmake_minimum_required(VERSION 3.24)
project(opencv)

set(CMAKE_CXX_STANDARD 17)

# set opencv
get_filename_component(ABSOLUTE_OpenCV_DIR ../../3rdparty/opencv ABSOLUTE)
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(OpenCV_DIR ${ABSOLUTE_OpenCV_DIR}/lib/cmake/opencv4)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(OpenCV_DIR ${ABSOLUTE_OpenCV_DIR})
ENDIF ()
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

add_executable(opencv main.cpp)
target_link_libraries(opencv ${OpenCV_LIBS})

add_executable(RotateRectCrop rotate_rect_crop.cpp)
target_link_libraries(RotateRectCrop ${OpenCV_LIBS})

add_executable(Jpeg2Mat jpeg2mat/main.cpp jpeg2mat/jpeg2mat.cpp)
target_link_libraries(Jpeg2Mat ${OpenCV_LIBS})

add_executable(ImgProcDetect imgproc_detect/imgproc_detect.cpp)
target_link_libraries(ImgProcDetect ${OpenCV_LIBS})

# Classify
add_executable(DNNClassify dnn_classify/main.cpp)
target_link_libraries(DNNClassify ${OpenCV_LIBS})

# YOLOv5
add_executable(yolo-image dnn_yolov5/yolo-image.cpp dnn_yolov5/src/yolodetector.cpp)
target_link_libraries(yolo-image ${OpenCV_LIBS})

add_executable(yolo-video dnn_yolov5/yolo-video.cpp dnn_yolov5/src/yolodetector.cpp)
target_link_libraries(yolo-video ${OpenCV_LIBS})

