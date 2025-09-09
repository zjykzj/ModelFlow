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
$ python3 samples/yolov5/infer.py --help
usage: infer.py [-h] [--backend {onnxruntime,tensorrt,triton}] [--processor {numpy,torch}] [--conf CONF] [--iou IOU] [--save_dir SAVE_DIR] weight source

YOLOv5 Inference

positional arguments:
  weight                Path to model file (e.g., yolov5s.onnx)
  source                Path to input image or video

optional arguments:
  -h, --help            show this help message and exit
  --backend {onnxruntime,tensorrt,triton}
                        Inference backend to use (default: onnxruntime)
  --processor {numpy,torch}
                        Pre/Post-processing backend: use NumPy or PyTorch (default: numpy)
  --conf CONF           Confidence threshold for object detection (default: 0.25)
  --iou IOU             IOU threshold for Non-Max Suppression (NMS) (default: 0.45)
  --save_dir SAVE_DIR   Output directory for results (default: output)
```

```shell
# Use NumPy for Pre/Post-processing
python3 samples/yolov5/infer.py models/yolov5s.onnx assets/bus.jpg

# Use PyTorch for Pre/Post-processing
python3 samples/yolov5/infer.py models/yolov5s.onnx assets/bus.jpg --processor torch

# Use TensorRT for infer
python3 samples/yolov5/infer.py models/yolov5s_fp16.engine assets --backend tensorrt
```