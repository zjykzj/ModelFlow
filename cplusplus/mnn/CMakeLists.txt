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

MESSAGE(STATUS ${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR})

#add_executable(det main.cpp)
add_executable(det infer.cpp)
target_link_libraries(det ${OpenCV_LIBS} ${INFER_LIB} ${EXT_LIB})


add_executable(mnn main.cpp)
