# YOLOv5

```text
Yolov5: https://github.com/ultralytics/yolov5
Commit id: 915bbf294bb74c859f0b41f1c23bc395014ea679
Tag: v7.0
```

## Usage

```shell
$ python3 samples/yolov5/infer.py --help
usage: infer.py [-h] [--backend {onnxruntime,tensorrt,triton}] [--processor {numpy,torch}] [--conf CONF] [--iou IOU] [--output OUTPUT] model input

YOLOv5 Inference

positional arguments:
  model                 Path to model file (e.g., yolov5s.onnx)
  input                 Path to input image or video

optional arguments:
  -h, --help            show this help message and exit
  --backend {onnxruntime,tensorrt,triton}
                        Inference backend to use (default: onnxruntime)
  --processor {numpy,torch}
                        Pre/Post-processing backend: use NumPy or PyTorch (default: numpy)
  --conf CONF           Confidence threshold for object detection (default: 0.25)
  --iou IOU             IOU threshold for Non-Max Suppression (NMS) (default: 0.45)
  --output OUTPUT       Output directory for results (default: output)
```

```shell
# Use NumPy for Pre/Post-processing
python infer.py yolov5s.onnx test.jpg --backend onnxruntime --processor numpy

# Use PyTorch for Pre/Post-processing
python infer.py yolov5s.onnx test.mp4 --backend onnxruntime --processor torch
```