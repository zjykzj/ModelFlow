# -*- coding: utf-8 -*-

"""
@date: 2023/3/20 下午5:21
@file: mnninfer.py
@author: zj
@description: 
"""

import MNN
import MNN.cv as cv
import MNN.numpy as np
import MNN.expr as expr

# 创建interpreter
interpreter = MNN.Interpreter("mobilenet_v1.mnn")
# 创建session
config = {}
config['precision'] = 'low'
config['backend'] = 'CPU'
config['thread'] = 4
session = interpreter.createSession(config)
# 获取会话的输入输出
input_tensor = interpreter.getSessionInput(session)
output_tensor = interpreter.getSessionOutput(session)

# 读取图片
image = cv.imread('cat.jpg')

dst_height = dst_width = 224
# 使用ImageProcess处理第一张图片，将图片转换为转换为size=(224, 224), dtype=float32，并赋值给input_data1
image_processer = MNN.CVImageProcess({'sourceFormat': MNN.CV_ImageFormat_BGR,
                                      'destFormat': MNN.CV_ImageFormat_BGR,
                                      'mean': (103.94, 116.78, 123.68, 0.0),
                                      'filterType': MNN.CV_Filter_BILINEAL,
                                      'normal': (0.017, 0.017, 0.017, 0.0)})
image_data = image.ptr
src_height, src_width, channel = image.shape
input_data1 = MNN.Tensor((1, dst_height, dst_width, channel), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Tensorflow)
#设置图像变换矩阵
matrix = MNN.CVMatrix()
x_scale = src_width / dst_width
y_scale = src_height / dst_height
matrix.setScale(x_scale, y_scale)
image_processer.setMatrix(matrix)
image_processer.convert(image_data, src_width, src_height, 0, input_data1)

# 使用cv模块处理第二张图片，将图片转换为转换为size=(224, 224), dtype=float32，并赋值给input_data2
image = cv.imread('TestMe.jpg')
image = cv.resize(image, (224, 224), mean=[103.94, 116.78, 123.68], norm=[0.017, 0.017, 0.017])
input_data2 = np.expand_dims(image, 0) # [224, 224, 3] -> [1, 224, 224, 3]

# 合并2张图片到，并赋值给input_data
input_data1 = expr.const(input_data1.getHost(), input_data1.getShape(), expr.NHWC) # Tensor -> Var
input_data = np.concatenate([input_data1, input_data2])  # [2, 224, 224, 3]
input_data = MNN.Tensor(input_data) # Var -> Tensor

# 演示多张图片输入，所以将输入resize到[2, 3, 224, 224]
interpreter.resizeTensor(input_tensor, (2, 3, 224, 224))
# 重新计算形状分配内存
interpreter.resizeSession(session)

# 拷贝数据到输入Tensor
input_tensor.copyFrom(input_data)

# 执行会话推理
interpreter.runSession(session)

# 从输出Tensor拷贝出数据
output_data = MNN.Tensor(output_tensor.getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
output_tensor.copyToHostTensor(output_data)

# 打印出分类结果: 282为猫，385为象
output_var = expr.const(output_data.getHost(), [2, 1001])
print("output belong to class: {}".format(np.argmax(output_var, 1)))
# output belong to class: array([282, 385], dtype=int32)