
# MNN

## Install

You should compile Infer Engine and Converter of MNN

* First，download MNN source code from [MNN Release](https://github.com/alibaba/MNN/releases/tag/1.2.0)

* Compile Infer Engine, see

1. [推理框架Linux / macOS编译](https://www.yuque.com/mnn/cn/build_linux)
2. [推理框架Windows编译](https://www.yuque.com/mnn/cn/build_windows)

* Compile Converter, see

1. [转换工具Linux / macOS编译](https://www.yuque.com/mnn/cn/cvrt_linux_mac)
2. [转换工具Windows编译](https://www.yuque.com/mnn/cn/cvrt_windows)

* Related links

1. [ setup.py: error: the following arguments are required: --version #1605 ](https://github.com/alibaba/MNN/issues/1605)

## Convert

```bash
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz
```

See: 

1. [模型转换](https://www.yuque.com/mnn/cn/model_convert)
2. [MNNConvert](https://github.com/alibaba/MNN/blob/master/tools/converter/README_CN.md)

## Test

MNN provides scripts to test whether different format conversion leads to inconsistent accuracy. 

1. cp `fastTestOnnx.py` to `build/`
2. cp `*.onnx` to `build/`
3. test: `python3 fastTestOnnx.py *.onnx` 

```bash
$ python3 fastTestOnnx.py mnist_cnn.onnx 


onnx/test.onnx
tensor(float)
['output']
inputs:
input
onnx/
outputs:
onnx/output.txt (1, 10)
onnx/
Test onnx
Start to Convert Other Model Format To MNN Model...
[15:39:15] /home/zj/mnn/MNN-1.2.0/tools/converter/source/onnx/onnxConverter.cpp:30: ONNX Model ir version: 7
Start to Optimize the MNN Net...
inputTensors : [ input, ]
outputTensors: [ output, ]
Converted Success!
input
output: output
TEST_SUCCESS
```

See: [正确性校验](https://www.yuque.com/mnn/cn/model_convert#nxImR)