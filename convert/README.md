
# README

1. pytorch -> onnx
2. onnx -> mnn
3. onnx -> mnn fp16
4. onnx -> mnn int8
5. mnn -> header file

## pytorch -> onnx

See `pytorch_to_onnx.py`

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

## onnx -> mnn fp16

```shell
./MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz --fp16
```

## onnx -> mnn int8

See https://mnn-docs.readthedocs.io/en/latest/tools/quant.html

```shell
./quantized.out mobilnet.mnn mobilnet_quant.mnn mobilnet_quant.json
```

## mnn -> header file

```shell
xxd -i xxx.mnn xxx.h
```