cmake_minimum_required(VERSION 3.20)
project(ONNX)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/googletest/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/googletest/lib)
set(gtest_lib libgmock.a libgmock_main.a libgtest.a libgtest_main.a)

add_executable(gtest_main ${CMAKE_SOURCE_DIR}/test/gtest_main.cc)
target_link_libraries(gtest_main ${gtest_lib} pthread)
