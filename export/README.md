# README

## yolov5 pt2onnx

```shell
python3 export.py --weights yolov5s.pt --include onnx --opset 12
```

## onnx2tensorrt

```shell
trtexec --onnx=efficientnet_b0.onnx --saveEngine=efficientnet_b0_fp16.engine --workspace=4096 --fp16

trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s_fp16.engine --workspace=4096 --fp16
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s_fp16.engine --workspace=4096 --fp16

trtexec --onnx=yolov8s-seg.onnx --saveEngine=yolov8s-seg_fp16.engine --workspace=4096 --fp16
```