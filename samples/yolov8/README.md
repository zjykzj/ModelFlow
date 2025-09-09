# YOLOv8

```shell
$ git clone https://github.com/ultralytics/ultralytics.git
$ cd ultralytics/
$ git checkout -b v8.2.103 v8.2.103

commit 25307552100e4c03c8fec7b0f7286b4244018e15 (HEAD -> v8.2.103, tag: v8.2.103)
Author: Glenn Jocher <glenn.jocher@ultralytics.com>
Date:   Sat Sep 28 20:31:21 2024 +0200
```

## Usage

```shell
$ python3 samples/yolov8/infer.py --help
usage: infer.py [-h] [--backend {onnxruntime,tensorrt,triton}] [--processor {numpy,torch}] [--conf CONF] [--iou IOU] [--save_dir SAVE_DIR] weight source

YOLOv8 Inference

positional arguments:
  weight                Path to model file (e.g., yolov8s.onnx)
  source                Path to input image or video

optional arguments:
  -h, --help            show this help message and exit
  --backend {onnxruntime,tensorrt,triton}
                        Inference backend to use (default: onnxruntime)
  --processor {numpy,torch}
                        Pre/Post-processing backend: use NumPy or PyTorch (default: numpy)
  --conf CONF           Confidence threshold for object detection (default: 0.25)
  --iou IOU             IOU threshold for Non-Max Suppression (NMS) (default: 0.7)
  --save_dir SAVE_DIR   Output directory for results (default: output)
```

```shell
# Use NumPy for Pre/Post-processing
python3 samples/yolov8/infer.py models/yolov8s.onnx assets/bus.jpg

# Use PyTorch for Pre/Post-processing
python3 samples/yolov8/infer.py models/yolov8s.onnx assets --processor torch

# Use TensorRT for infer
python3 samples/yolov8/infer.py models/yolov8s_fp16.engine assets --backend tensorrt
```