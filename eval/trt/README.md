
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

## Device Info

```text
[02/22/2026-12:31:53] [I] === Device Information ===
[02/22/2026-12:31:53] [I] Selected Device: NVIDIA GeForce RTX 4060 Laptop GPU
[02/22/2026-12:31:53] [I] Compute Capability: 8.9
[02/22/2026-12:31:53] [I] SMs: 24
[02/22/2026-12:31:53] [I] Compute Clock Rate: 1.89 GHz
[02/22/2026-12:31:53] [I] Device Global Memory: 8187 MiB
[02/22/2026-12:31:53] [I] Shared Memory per SM: 100 KiB
[02/22/2026-12:31:53] [I] Memory Bus Width: 128 bits (ECC disabled)
[02/22/2026-12:31:53] [I] Memory Clock Rate: 8.001 GHz
[02/22/2026-12:31:53] [I] 
[02/22/2026-12:31:53] [I] TensorRT version: 8.5.0
[02/22/2026-12:31:53] [I] [TRT] [MemUsageChange] Init CUDA: CPU +13, GPU +0, now: CPU 31, GPU 1088 (MiB)
[02/22/2026-12:31:55] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +524, GPU +114, now: CPU 608, GPU 1202 (MiB)
[02/22/2026-12:31:55] [I] Start parsing network model
[02/22/2026-12:31:55] [I] [TRT] ----------------------------------------------------------------
[02/22/2026-12:31:55] [I] [TRT] Input filename:   efficientnet_b0.onnx
[02/22/2026-12:31:55] [I] [TRT] ONNX IR version:  0.0.7
[02/22/2026-12:31:55] [I] [TRT] Opset version:    12
[02/22/2026-12:31:55] [I] [TRT] Producer name:    pytorch
[02/22/2026-12:31:55] [I] [TRT] Producer version: 1.13.1
[02/22/2026-12:31:55] [I] [TRT] Domain:           
[02/22/2026-12:31:55] [I] [TRT] Model version:    0
[02/22/2026-12:31:55] [I] [TRT] Doc string:       
[02/22/2026-12:31:55] [I] [TRT] ----------------------------------------------------------------
[02/22/2026-12:31:55] [I] Finish parsing network model
[02/22/2026-12:31:55] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +6, GPU +12, now: CPU 636, GPU 1214 (MiB)
[02/22/2026-12:31:55] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +2, GPU +8, now: CPU 638, GPU 1222 (MiB)
```