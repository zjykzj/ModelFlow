# README

## Container

```text
$ docker run --gpus=all -it -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd):/workdir --workdir=/workdir nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=./models/triton/

=============================
== Triton Inference Server ==
=============================

NVIDIA Release 23.10 (build 72127154)
Triton Server Version 2.39.0

Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

I0223 08:55:17.236225 1 pinned_memory_manager.cc:241] Pinned memory pool is created at '0x204c00000' with size 268435456
I0223 08:55:17.236367 1 cuda_memory_manager.cc:107] CUDA memory pool is created on device 0 with size 67108864
I0223 08:55:17.238655 1 model_lifecycle.cc:461] loading: Detect_COCO_YOLOv5s_TensorRT:1
I0223 08:55:17.238711 1 model_lifecycle.cc:461] loading: Detect_COCO_YOLOv8s_ONNX:1
I0223 08:55:17.238747 1 model_lifecycle.cc:461] loading: Classify_ImageNet_EfficientNetB0_ONNX:1
I0223 08:55:17.238772 1 model_lifecycle.cc:461] loading: Detect_COCO_YOLOv5s_ONNX:1
I0223 08:55:17.238802 1 model_lifecycle.cc:461] loading: Classify_ImageNet_EfficientNetB0_TensorRT:1
I0223 08:55:17.238814 1 model_lifecycle.cc:461] loading: Detect_COCO_YOLOv8s_TensorRT:1
I0223 08:55:17.238825 1 model_lifecycle.cc:461] loading: Segment_COCO_YOLOv8sSeg_ONNX:1
I0223 08:55:17.238845 1 model_lifecycle.cc:461] loading: Segment_COCO_YOLOv8sSeg_TensorRT:1
I0223 08:55:17.285936 1 tensorrt.cc:65] TRITONBACKEND_Initialize: tensorrt
I0223 08:55:17.285987 1 tensorrt.cc:75] Triton TRITONBACKEND API version: 1.16
I0223 08:55:17.285989 1 tensorrt.cc:81] 'tensorrt' TRITONBACKEND API version: 1.16
I0223 08:55:17.285992 1 tensorrt.cc:105] backend configuration:
{"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
I0223 08:55:17.286128 1 tensorrt.cc:231] TRITONBACKEND_ModelInitialize: Detect_COCO_YOLOv5s_TensorRT (version 1)
I0223 08:55:17.286940 1 onnxruntime.cc:2608] TRITONBACKEND_Initialize: onnxruntime
I0223 08:55:17.286964 1 onnxruntime.cc:2618] Triton TRITONBACKEND API version: 1.16
I0223 08:55:17.286967 1 onnxruntime.cc:2624] 'onnxruntime' TRITONBACKEND API version: 1.16
I0223 08:55:17.286970 1 onnxruntime.cc:2654] backend configuration:
{"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","min-compute-capability":"6.000000","default-max-batch-size":"4"}}
I0223 08:55:17.298234 1 onnxruntime.cc:2719] TRITONBACKEND_ModelInitialize: Detect_COCO_YOLOv8s_ONNX (version 1)
I0223 08:55:17.298305 1 onnxruntime.cc:2719] TRITONBACKEND_ModelInitialize: Classify_ImageNet_EfficientNetB0_ONNX (version 1)
I0223 08:55:17.298323 1 onnxruntime.cc:2719] TRITONBACKEND_ModelInitialize: Detect_COCO_YOLOv5s_ONNX (version 1)
I0223 08:55:17.298576 1 onnxruntime.cc:692] skipping model configuration auto-complete for 'Detect_COCO_YOLOv5s_ONNX': inputs and outputs already specified
I0223 08:55:17.298576 1 onnxruntime.cc:692] skipping model configuration auto-complete for 'Classify_ImageNet_EfficientNetB0_ONNX': inputs and outputs already specified
I0223 08:55:17.298855 1 onnxruntime.cc:692] skipping model configuration auto-complete for 'Detect_COCO_YOLOv8s_ONNX': inputs and outputs already specified
I0223 08:55:17.299087 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: Detect_COCO_YOLOv5s_ONNX_0 (GPU device 0)
I0223 08:55:17.299342 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: Classify_ImageNet_EfficientNetB0_ONNX_0 (GPU device 0)
I0223 08:55:17.300529 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: Detect_COCO_YOLOv8s_ONNX_0 (GPU device 0)
I0223 08:55:17.316635 1 logging.cc:46] Loaded engine size: 16 MiB
I0223 08:55:17.369536 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +13, now: CPU 0, GPU 13 (MiB)
W0223 08:55:17.370444 1 model_state.cc:530] The specified dimensions in model config for Detect_COCO_YOLOv5s_TensorRT hints that batching is unavailable
I0223 08:55:17.377545 1 tensorrt.cc:297] TRITONBACKEND_ModelInstanceInitialize: Detect_COCO_YOLOv5s_TensorRT_0 (GPU device 0)
I0223 08:55:17.394652 1 logging.cc:46] Loaded engine size: 16 MiB
I0223 08:55:17.436343 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +13, now: CPU 0, GPU 13 (MiB)
I0223 08:55:17.438737 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +30, now: CPU 0, GPU 43 (MiB)
I0223 08:55:17.439936 1 instance_state.cc:188] Created instance Detect_COCO_YOLOv5s_TensorRT_0 on GPU 0 with stream priority 0 and optimization profile default[0];
I0223 08:55:17.440526 1 model_lifecycle.cc:818] successfully loaded 'Detect_COCO_YOLOv5s_TensorRT'
I0223 08:55:17.440615 1 tensorrt.cc:231] TRITONBACKEND_ModelInitialize: Classify_ImageNet_EfficientNetB0_TensorRT (version 1)
I0223 08:55:17.441041 1 logging.cc:46] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
I0223 08:55:17.442975 1 model_lifecycle.cc:818] successfully loaded 'Detect_COCO_YOLOv5s_ONNX'
I0223 08:55:17.443058 1 tensorrt.cc:231] TRITONBACKEND_ModelInitialize: Detect_COCO_YOLOv8s_TensorRT (version 1)
I0223 08:55:17.443317 1 logging.cc:46] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
I0223 08:55:17.461201 1 logging.cc:46] Loaded engine size: 18 MiB
I0223 08:55:17.478478 1 logging.cc:46] Loaded engine size: 24 MiB
I0223 08:55:17.527869 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +36, now: CPU 0, GPU 79 (MiB)
W0223 08:55:17.528469 1 model_state.cc:530] The specified dimensions in model config for Classify_ImageNet_EfficientNetB0_TensorRT hints that batching is unavailable
I0223 08:55:17.542000 1 tensorrt.cc:297] TRITONBACKEND_ModelInstanceInitialize: Classify_ImageNet_EfficientNetB0_TensorRT_0 (GPU device 0)
I0223 08:55:17.546026 1 logging.cc:46] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
I0223 08:55:17.547813 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +7, now: CPU 0, GPU 65 (MiB)
W0223 08:55:17.549412 1 model_state.cc:530] The specified dimensions in model config for Detect_COCO_YOLOv8s_TensorRT hints that batching is unavailable
I0223 08:55:17.557250 1 tensorrt.cc:297] TRITONBACKEND_ModelInstanceInitialize: Detect_COCO_YOLOv8s_TensorRT_0 (GPU device 0)
I0223 08:55:17.558207 1 logging.cc:46] Loaded engine size: 18 MiB
I0223 08:55:17.594273 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +15, now: CPU 0, GPU 58 (MiB)
I0223 08:55:17.594972 1 logging.cc:46] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
I0223 08:55:17.595916 1 model_lifecycle.cc:818] successfully loaded 'Detect_COCO_YOLOv8s_ONNX'
I0223 08:55:17.596005 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +4, now: CPU 1, GPU 62 (MiB)
I0223 08:55:17.596013 1 onnxruntime.cc:2719] TRITONBACKEND_ModelInitialize: Segment_COCO_YOLOv8sSeg_ONNX (version 1)
I0223 08:55:17.596321 1 instance_state.cc:188] Created instance Classify_ImageNet_EfficientNetB0_TensorRT_0 on GPU 0 with stream priority 0 and optimization profile default[0];
I0223 08:55:17.596364 1 onnxruntime.cc:692] skipping model configuration auto-complete for 'Segment_COCO_YOLOv8sSeg_ONNX': inputs and outputs already specified
I0223 08:55:17.596915 1 model_lifecycle.cc:818] successfully loaded 'Classify_ImageNet_EfficientNetB0_TensorRT'
I0223 08:55:17.597081 1 tensorrt.cc:231] TRITONBACKEND_ModelInitialize: Segment_COCO_YOLOv8sSeg_TensorRT (version 1)
I0223 08:55:17.597119 1 onnxruntime.cc:2784] TRITONBACKEND_ModelInstanceInitialize: Segment_COCO_YOLOv8sSeg_ONNX_0 (GPU device 0)
I0223 08:55:17.597633 1 logging.cc:46] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
I0223 08:55:17.626154 1 logging.cc:46] Loaded engine size: 24 MiB
I0223 08:55:17.635587 1 logging.cc:46] Loaded engine size: 25 MiB
I0223 08:55:17.697199 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +45, now: CPU 1, GPU 107 (MiB)
I0223 08:55:17.698254 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +24, now: CPU 1, GPU 107 (MiB)
W0223 08:55:17.699589 1 model_state.cc:530] The specified dimensions in model config for Segment_COCO_YOLOv8sSeg_TensorRT hints that batching is unavailable
I0223 08:55:17.700244 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +30, now: CPU 2, GPU 137 (MiB)
I0223 08:55:17.701609 1 instance_state.cc:188] Created instance Detect_COCO_YOLOv8s_TensorRT_0 on GPU 0 with stream priority 0 and optimization profile default[0];
I0223 08:55:17.701901 1 model_lifecycle.cc:818] successfully loaded 'Detect_COCO_YOLOv8s_TensorRT'
I0223 08:55:17.709735 1 tensorrt.cc:297] TRITONBACKEND_ModelInstanceInitialize: Segment_COCO_YOLOv8sSeg_TensorRT_0 (GPU device 0)
I0223 08:55:17.712845 1 logging.cc:46] The logger passed into createInferRuntime differs from one already provided for an existing builder, runtime, or refitter. Uses of the global logger, returned by nvinfer1::getLogger(), will return the existing value.
I0223 08:55:17.717214 1 model_lifecycle.cc:818] successfully loaded 'Classify_ImageNet_EfficientNetB0_ONNX'
I0223 08:55:17.741071 1 logging.cc:46] Loaded engine size: 25 MiB
I0223 08:55:17.773028 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +1, GPU +23, now: CPU 2, GPU 138 (MiB)
I0223 08:55:17.776203 1 logging.cc:46] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +34, now: CPU 2, GPU 172 (MiB)
I0223 08:55:17.777539 1 instance_state.cc:188] Created instance Segment_COCO_YOLOv8sSeg_TensorRT_0 on GPU 0 with stream priority 0 and optimization profile default[0];
I0223 08:55:17.777817 1 model_lifecycle.cc:818] successfully loaded 'Segment_COCO_YOLOv8sSeg_TensorRT'
I0223 08:55:17.808319 1 model_lifecycle.cc:818] successfully loaded 'Segment_COCO_YOLOv8sSeg_ONNX'
I0223 08:55:17.808430 1 server.cc:592]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I0223 08:55:17.808481 1 server.cc:619]
+-------------+-----------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| Backend     | Path                                                            | Config                                                                                         |
+-------------+-----------------------------------------------------------------+------------------------------------------------------------------------------------------------+
| tensorrt    | /opt/tritonserver/backends/tensorrt/libtriton_tensorrt.so       | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","mi |
|             |                                                                 | n-compute-capability":"6.000000","default-max-batch-size":"4"}}                                |
| onnxruntime | /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | {"cmdline":{"auto-complete-config":"true","backend-directory":"/opt/tritonserver/backends","mi |
|             |                                                                 | n-compute-capability":"6.000000","default-max-batch-size":"4"}}                                |
+-------------+-----------------------------------------------------------------+------------------------------------------------------------------------------------------------+

I0223 08:55:17.808531 1 server.cc:662]
+-------------------------------------------+---------+--------+
| Model                                     | Version | Status |
+-------------------------------------------+---------+--------+
| Classify_ImageNet_EfficientNetB0_ONNX     | 1       | READY  |
| Classify_ImageNet_EfficientNetB0_TensorRT | 1       | READY  |
| Detect_COCO_YOLOv5s_ONNX                  | 1       | READY  |
| Detect_COCO_YOLOv5s_TensorRT              | 1       | READY  |
| Detect_COCO_YOLOv8s_ONNX                  | 1       | READY  |
| Detect_COCO_YOLOv8s_TensorRT              | 1       | READY  |
| Segment_COCO_YOLOv8sSeg_ONNX              | 1       | READY  |
| Segment_COCO_YOLOv8sSeg_TensorRT          | 1       | READY  |
+-------------------------------------------+---------+--------+

I0223 08:55:17.832636 1 metrics.cc:817] Collecting metrics for GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU
I0223 08:55:17.832879 1 metrics.cc:710] Collecting CPU metrics
I0223 08:55:17.833029 1 tritonserver.cc:2458]
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                                        |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                                       |
| server_version                   | 2.39.0                                                                                                                                       |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_shared_memory cuda_s |
|                                  | hared_memory binary_tensor_data parameters statistics trace logging                                                                          |
| model_repository_path[0]         | ./models/triton/                                                                                                                             |
| model_control_mode               | MODE_NONE                                                                                                                                    |
| strict_model_config              | 0                                                                                                                                            |
| rate_limit                       | OFF                                                                                                                                          |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                                    |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                                     |
| min_supported_compute_capability | 6.0                                                                                                                                          |
| strict_readiness                 | 1                                                                                                                                            |
| exit_timeout                     | 30                                                                                                                                           |
| cache_enabled                    | 0                                                                                                                                            |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------+

I0223 08:55:17.841990 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I0223 08:55:17.842271 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I0223 08:55:17.884119 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
W0223 08:55:18.836390 1 metrics.cc:582] Unable to get power limit for GPU 0. Status:Success, value:0.000000
W0223 08:55:19.842893 1 metrics.cc:582] Unable to get power limit for GPU 0. Status:Success, value:0.000000
W0223 08:55:20.846790 1 metrics.cc:582] Unable to get power limit for GPU 0. Status:Success, value:0.000000
```

## Check

```text
# HTTP 健康检查
curl -v http://localhost:8000/v2/health/ready
```