# -*- coding: utf-8 -*-

"""
@Time    : 2023/10/25 16:14
@File    : pth2onnx2trt_r50.py
@Author  : zj
@Description: Full reference https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
"""

import torch
import torch.onnx
import torchvision.models as models
from torchvision.transforms import Normalize

from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np

BATCH_SIZE = 32
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)

USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32


def get_input_batch(url='../assets/retriever-golden/n02099601_3004.jpg'):
    # url = 'https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'
    # [?, ?] -> [224, 224]
    img = resize(io.imread(url), (224, 224))
    # [224, 224, 3] -> [1, 224, 224, 3]
    img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)  # Expand image to have a batch dimension
    # [1, 224, 224, 3] -> [BATCH_SIZE, 224, 224, 3]
    input_batch = np.array(np.repeat(img, BATCH_SIZE, axis=0), dtype=np.float32)  # Repeat across the batch dimension
    print(f"input_batch: {input_batch.shape}")

    return input_batch


def step1_test_with_pytorch():
    input_batch = get_input_batch()
    plt.imshow(input_batch[0].astype(np.float32))
    plt.savefig("dog.jpg")

    resnet50_gpu = models.resnet50(pretrained=True, progress=False).to("cuda").eval()
    # [N, H, W, C] -> [N, C, W, H] -> [N, C, H, W]
    input_batch_chw = torch.from_numpy(input_batch).transpose(1, 3).transpose(2, 3)
    input_batch_gpu = input_batch_chw.to("cuda")
    print(f"input_batch_gpu: {input_batch_gpu.shape}")

    with torch.no_grad():
        predictions = np.array(resnet50_gpu(input_batch_gpu).cpu())
    print(f"prediction shape: {predictions.shape}")

    indices = (-predictions[0]).argsort()[:5]
    print("Class | Likelihood")
    print(list(zip(indices, predictions[0][indices])))

    return resnet50_gpu, input_batch_gpu, input_batch


def step2_verify_fp16_performance(resnet50_gpu, input_batch_gpu):
    resnet50_gpu_half = resnet50_gpu.half()
    input_half = input_batch_gpu.half()

    with torch.no_grad():
        preds_fp16 = np.array(resnet50_gpu_half(input_half).cpu())  # Warm Up
    print(f"preds_fp16.shape: {preds_fp16.shape}")

    indices = (-preds_fp16[0]).argsort()[:5]
    print("Class | Likelihood")
    print(list(zip(indices, preds_fp16[0][indices])))


def step3_export_onnx():
    # load the pretrained model
    resnet50 = models.resnet50(pretrained=True, progress=False).eval()

    # export the model to ONNX
    torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch.onnx", verbose=False)


def step5_test_with_tensorrt(input_batch, trt_path="resnet_engine_pytorch.trt", USE_FP16=False):
    target_dtype = np.float16 if USE_FP16 else np.float32

    def preprocess_image(img, dtype=np.float16):
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        result = norm(torch.from_numpy(img).transpose(0, 2).transpose(1, 2))
        return np.array(result, dtype=dtype)

    preprocessed_images = np.array([preprocess_image(image, dtype=target_dtype) for image in input_batch])

    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    def load_trt_runtime(trt_path):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(trt_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        return context

    context = load_trt_runtime(trt_path)

    # allocate input and output memory
    # need to set input and output precisions to FP16 to fully enable it
    output = np.empty([BATCH_SIZE, 1000], dtype=target_dtype)

    # allocate device memory
    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict(batch):  # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()

        return output

    print("Warming up...")
    pred = predict(preprocessed_images)
    print("Done warming up!")
    pred = predict(preprocessed_images)

    indices = (-pred[0]).argsort()[:5]
    print("Class | Probability (out of 1)")
    print(list(zip(indices, pred[0][indices])))


if __name__ == '__main__':
    print("=> step1_test_with_pytorch")
    resnet50_gpu, input_batch_gpu, input_batch = step1_test_with_pytorch()
    print("=> step2_verify_fp16_performance")
    step2_verify_fp16_performance(resnet50_gpu, input_batch_gpu)
    # print("=> step3_export_onnx")
    # step3_export_onnx()
    # print("=> step4_export_tensorrt")
    #
    # # Float32
    # trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch.trt  --explicitBatch
    # # FP16
    # trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet_engine_pytorch_fp16.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
    #
    print("=> step5_test_with_tensorrt")
    step5_test_with_tensorrt(input_batch, trt_path="resnet_engine_pytorch.trt", USE_FP16=False)
    print("=> step6_verify_tensorrt_with_fp16")
    step5_test_with_tensorrt(input_batch, trt_path="resnet_engine_pytorch_fp16.trt", USE_FP16=True)
