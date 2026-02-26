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

Specially for tritonserver

```text
docker run --gpus=all -it -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3 bash

/usr/src/tensorrt/bin/trtexec --onnx=efficientnet_b0.onnx --saveEngine=model.plan --workspace=4096 --fp16

/usr/src/tensorrt/bin/trtexec --onnx=yolov5s.onnx --saveEngine=model.plan --workspace=4096 --fp16
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=model.plan --workspace=4096 --fp16

/usr/src/tensorrt/bin/trtexec --onnx=yolov8s-seg.onnx --saveEngine=model.plan --workspace=4096 --fp16
```

### int8

Create int8 calibrated cache

```shell
python3 export/scripts/generate_calib_cache_for_imagenet.py --input_dir export/cal_imagenet_src --output_dir export/cal_imagenet_dst

python3 export/safe_int8_build_by_torch.py --onnx export/yolov5s.onnx --calib_dir export/coco_calib_dst --output yolov5s_int8.engine --input_shape 1 3 640 640
```

onnx2int8

```shell
python3 export/safe_int8_build_by_torch.py --onnx export/efficientnet_b0.onnx --calib_dir export/cal_imagenet_dst --output efficientnet_b0_int8.engine --input_shape 1 3 224 224

python3 export/safe_int8_build_by_torch.py --onnx export/yolov5s.onnx --calib_dir export/coco_calib_dst --output yolov5s_int8.engine --input_shape 1 3 640 640
```

### Test

```text
trtexec --loadEngine=yolov5s_slim_fp16.engine --iterations=100
```

Specially in AutoDL

```text
# 1. 查找库文件位置
# find /root -name "libnvinfer_plugin.so.8" 2>/dev/null
/root/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so.8
# 2. 设置环境变量 (将 TensorRT lib 目录加入路径)
export LD_LIBRARY_PATH=/root/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
# 3. 再次运行测试命令
trtexec --loadEngine=yolov5s_int8_final.engine --iterations=10
```


