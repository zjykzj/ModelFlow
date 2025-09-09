# YOLOv5

```shell
$ git clone https://github.com/ultralytics/yolov5.git
$ cd yolov5/
$ git checkout -b v7.0 v7.0

commit 915bbf294bb74c859f0b41f1c23bc395014ea679 (HEAD -> v7.0, tag: v7.0)
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Tue Nov 22 16:23:47 2022 +0100
```

## Usage

```shell
# Use NumPy for Pre/Post-processing
python3 samples/yolov5/infer.py models/yolov5s.onnx assets/bus.jpg --data core/cfgs/coco.yaml

# Use PyTorch for Pre/Post-processing
python3 samples/yolov5/infer.py models/yolov5s.onnx assets/bus.jpg --data core/cfgs/coco.yaml --processor torch

# Use TensorRT for infer
python3 samples/yolov5/infer.py models/yolov5s_fp16.engine assets --data core/cfgs/coco.yaml --backend tensorrt
```