
# README

1. pytorch -> onnx
2. onnx -> mnn
   1. onnx -> mnn fp16
   2. onnx -> mnn int8
   3. mnn -> header file
3. onnx -> tensorrt

## pytorch -> onnx

See `pytorch_to_onnx.py`

固定轴转换 + 测试

静态轴转换 + 测试

## onnx -> mnn

See https://mnn-docs.readthedocs.io/en/latest/tools/convert.html

* First, check onnx is fine

```shell
python3 fastTestOnnx.py xxx.onnx
```

* Then, convert onnx to model

```shell
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz
```

### onnx -> mnn fp16

```shell
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz --fp16
```

### onnx -> mnn int8

See https://mnn-docs.readthedocs.io/en/latest/tools/quant.html

```shell
./quantized.out mobilnet.mnn mobilnet_quant.mnn mobilnet_quant.json
```

### mnn -> header file

```shell
xxd -i xxx.mnn xxx.h
```

## onnx -> tensorrt

See 

1. https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#onnx-export
2. https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#ex-deploy-onnx

```shell
# 固定批量大小
trtexec --onnx=resnet18_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch
# 半精度转换
trtexec --onnx=resnet18_pytorch.onnx --saveEngine=resnet_engine_pytorch_fp16.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
```