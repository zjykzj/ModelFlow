
# README

主要划分为两个部分：

1. numpy实现
2. pytorch实现

另外，根据不同的推理后端，可以分为：

1. pytorch实现
2. onnxruntime实现
3. tensorrt实现
4. triton实现

注意：在这个文件夹的yolov5实现中，参考官方仓库进行图像预处理


关于关键接口的数据输入和返回，尽量和官方仓库实现保持一致，提高复用性。

NMS（non-maximum-suppression）：基本操作流程

YOLOv5的后处理和YOLOv8的后处理会有差别？比如置信度？

这个文件夹里面实现的模块化的YOLO算法，主要是YOLOv5和YOLOv8的模块化推理实现，划分为3层：

1. 前处理
   2. 图像预处理：采用opencv算法实现
      3. 选择一：图像缩放
      4. 选择二：图像缩放+填充 
   5. 数据预处理 
      6. 选择一：pytorch 
      7. 选择二：numpy
2. 模型推理
   3. 选择一：Pytorch
   4. 选择二：Onnxruntime
   5. 选择三：TensorRT
   6. 选择四：Triton
3. 后处理
   4. 数据后处理
      5. 选择一：pytorch
      6. 选择二：numpy

批量运行也好，单张运行也好，都支持
先不考虑：应该给出测试代码，比如验证coco数据集，支持单张或者批量的测试结果，查看精度、召回率和mAP。

边缘端就是单张，不考虑批量结果。关于测试代码，先不写，尽快推进yolov8的前后处理代码整理

之后就是yolov5和yolov8网络的定义，以及yolov5损失函数/yolov8损失函数的解析。