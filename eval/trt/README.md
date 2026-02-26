
# README

## Container

```text
(py39) zjykzj@LAPTOP-S3BIPLGN:~/myai/ModelFlow$ docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/zjykzj:/workdir --workdir=/workdir --name ultra ultralytics/yolov5:v7.0 bash

=============
== PyTorch ==
=============

NVIDIA Release 22.10 (build 46164382)
PyTorch Version 1.13.0a0+d0d6b1f

Container image Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Copyright (c) 2014-2022 Facebook Inc.
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
Copyright (c) 2015      Google Inc.
Copyright (c) 2015      Yangqing Jia
Copyright (c) 2013-2016 The Caffe contributors
All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

root@d27b9418abdf:/workdir# 
```

## Environment

```text
# 执行
pip3 install pycuda==2024.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证 PyCUDA
python3 -c "import pycuda.driver as cuda; import pycuda.autoinit; print(f'✅ PyCUDA: {cuda.Device.count()} GPUs')"
```