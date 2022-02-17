cmake_minimum_required(VERSION 3.20)
project(mnn_infer)

set(CMAKE_CXX_STANDARD 11)

include_directories(/home/zj/mnn/MNN-1.2.0/include)
link_directories("/home/zj/mnn/MNN-1.2.0/build")

add_executable(mnn_infer main.cpp infer_engine.cpp)
target_link_libraries(mnn_infer libMNN.so)

add_executable(mnn_norm mnn_norm.cpp)
target_link_libraries(mnn_norm libMNN.so)

add_executable(mnn_resize mnn_resize.cpp)
target_link_libraries(mnn_resize libMNN.so)