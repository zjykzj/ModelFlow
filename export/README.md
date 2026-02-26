# README

## yolov5 pt2onnx

```shell
python3 export.py --weights yolov5s.pt --include onnx --opset 12
```

## onnx2tensorrt

### fp16

```shell
trtexec --onnx=efficientnet_b0.onnx --saveEngine=efficientnet_b0_fp16.engine --workspace=4096 --fp16

trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp16.engine --workspace=4096 --fp16
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s_fp16.engine --workspace=4096 --fp16

trtexec --onnx=yolov8s-seg.onnx --saveEngine=yolov8s-seg_fp16.engine --workspace=4096 --fp16
```

### For tritonserver

```text
docker run --gpus=all -it -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3 bash

/usr/src/tensorrt/bin/trtexec --onnx=efficientnet_b0.onnx --saveEngine=model.plan --workspace=4096 --fp16

/usr/src/tensorrt/bin/trtexec --onnx=yolov5s.onnx --saveEngine=model.plan --workspace=4096 --fp16
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=model.plan --workspace=4096 --fp16

/usr/src/tensorrt/bin/trtexec --onnx=yolov8s-seg.onnx --saveEngine=model.plan --workspace=4096 --fp16
```

### Test

```text
trtexec --loadEngine=yolov5s_slim_fp16.engine --iterations=100
```