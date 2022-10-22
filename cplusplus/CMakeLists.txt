cmake_minimum_required(VERSION 3.20)
project(ONNX)

set(CMAKE_CXX_STANDARD 17)

if (NOT BUILD_LIB_NAME)
    set(BUILD_LIB_NAME linux_mnn)
endif ()
if (NOT INFER_BACKEND)
    set(INFER_BACKEND mnn)
endif ()
if (NOT BUILD_LIB_ABI)
    set(BUILD_LIB_ABI ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif ()

# Host info
MESSAGE(STATUS "${CMAKE_HOST_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-${CMAKE_HOST_SYSTEM_VERSION}")

# Compile info
MESSAGE(STATUS "${BUILD_LIB_NAME}")
MESSAGE(STATUS "${INFER_BACKEND}")
MESSAGE(STATUS "${BUILD_LIB_ABI}")

# 3rdparty
if (ANDROID)
    MESSAGE(STATUS "Using Android ${BUILD_LIB_ABI}")
    set(EXT_LIBS log)
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "Using Linux ${BUILD_LIB_ABI}")
    set(EXT_LIBS pthread)
endif ()

set(INFER_ROOT ${CMAKE_SOURCE_DIR}/3rdparty/${INFER_BACKEND})
include_directories(${INFER_ROOT}/include)
link_directories(${INFER_ROOT}/libs/${BUILD_LIB_ABI}/)

if (${INFER_BACKEND} MATCHES mnn)
    set(INFER_LIBS libMNN.so)
endif ()

# Project
include_directories(${CMAKE_SOURCE_DIR}/include/)
set(SRC ${CMAKE_SOURCE_DIR}/source/recoginterface.cc)

include_directories(${CMAKE_SOURCE_DIR}/infer/)
if (${INFER_BACKEND} MATCHES MNN)
    include_directories(${CMAKE_SOURCE_DIR}/infer/mnn/)
    set(INFER_SRC
            ${CMAKE_SOURCE_DIR}/infer/mnn/inferimpl.cc)
endif ()

MESSAGE(STATUS "Compile ${BUILD_LIB_NAME}")
add_library(${BUILD_LIB_NAME} ${SRC} ${INFER_SRC})
target_link_libraries(${BUILD_LIB_NAME}
        ${INFER_LIBS}
        ${EXT_LIBS}
        )

# Unit test
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/googletest/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/googletest/lib)
set(gtest_lib libgmock.a libgmock_main.a libgtest.a libgtest_main.a)

MESSAGE(STATUS "Compile gtest_main")
add_executable(gtest_main ${CMAKE_SOURCE_DIR}/test/gtest_main.cc)
target_link_libraries(gtest_main ${gtest_lib} pthread)
