cmake_minimum_required(VERSION 3.20)
project(zco)

set(CMAKE_CXX_STANDARD 14)


# set opencv
# see [get_filename_component](https://cmake.org/cmake/help/latest/command/get_filename_component.html)
get_filename_component(ABSOLUTE_OpenCV_DIR ../../3rdparty/opencv ABSOLUTE)
# Add the installation prefix of "OpenCV" to CMAKE_PREFIX_PATH or set
# "OpenCV_DIR" to a directory containing OpenCVConfig.cmake/opencv-config.cmake.
#set(OpenCV_DIR ../../3rdparty/opencv/lib/cmake/opencv4)
set(OpenCV_DIR ${ABSOLUTE_OpenCV_DIR}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

# see [再谈CMake与RPATH](https://www.cnblogs.com/rickyk/p/3884257.html)
#set(CMAKE_INSTALL_RPATH ${CMAKE_PREFIX_PATH}/lib)

# make library
add_library(zco SHARED
        source/library.cpp
        source/library.h
        source/image_process.cpp
        source/image_process.h
        source/common_operation.cpp
        source/common_operation.h)
target_link_libraries(zco ${OpenCV_LIBS})
set_target_properties(zco PROPERTIES INSTALL_RPATH "${ABSOLUTE_OpenCV_DIR}/lib")

# make executable
add_executable(test_library
        test/test_library.cpp
        source/library.cpp
        source/library.h)

# make executable
add_executable(test_image_process
        test/test_image_process.cpp
        source/image_process.h
        source/image_process.cpp
        )
target_link_libraries(test_image_process ${OpenCV_LIBS})
set_target_properties(test_image_process PROPERTIES INSTALL_RPATH "${ABSOLUTE_OpenCV_DIR}/lib")

# install
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
install(TARGETS zco LIBRARY DESTINATION lib)
install(TARGETS test_library RUNTIME DESTINATION bin)
install(TARGETS test_image_process RUNTIME DESTINATION bin)
install(FILES
        source/library.h
        source/image_process.h
        source/common_operation.h
        DESTINATION include)