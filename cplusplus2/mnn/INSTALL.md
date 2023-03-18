
# 安装

## 下载

* [https://github.com/alibaba/MNN/releases](https://github.com/alibaba/MNN/releases)

## 编译

* [编译宏介绍](https://mnn-docs.readthedocs.io/en/latest/compile/cmake.html)
* [主库编译](https://mnn-docs.readthedocs.io/en/latest/compile/engine.html)

### Linux

```shell
cd /path/to/MNN
./schema/generate.sh
./tools/script/get_model.sh # 可选，模型仅demo工程需要

# 动态库
mkdir build && cd build && cmake .. && make -j8
# 静态库
cmake -DMNN_BUILD_SHARED_LIBS=OFF .. && make -j8
```

### Android

```shell
export ANDROID_NDK=/Users/username/path/to/android-ndk-r14b

cd /path/to/MNN
cd project/android
# 编译armv7动态库
mkdir build_32 && cd build_32 && ../build_32.sh
# 编译armv8动态库
mkdir build_64 && cd build_64 && ../build_64.sh
```

* [Android 静态库编译 #619](https://github.com/alibaba/MNN/issues/619)
* [libMNN.a 有151M #521](https://github.com/alibaba/MNN/issues/521)

### AArch64

```shell
mkdir -p linaro/aarch64
cd linaro/aarch64
wget https://releases.linaro.org/components/toolchain/binaries/latest-7/arm-linux-gnueabi/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabi.tar.xz
tar xvf gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabi.tar.xz

export cross_compile_toolchain=linaro/aarch64
mkdir build && cd build
cmake .. \
-DCMAKE_SYSTEM_NAME=Linux \
-DCMAKE_SYSTEM_VERSION=1 \
-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
-DCMAKE_C_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=$cross_compile_toolchain/bin/aarch64-linux-gnu-g++
make -j4
```