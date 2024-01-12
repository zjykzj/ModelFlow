# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/16 14:32
@File    : yolov8_trt_w_torch.py
@Author  : zj
@Description:

# Start Docker Container
>>>docker run --gpus all -it --rm -v ${PWD}:/workdir --workdir=/workdir ultralytics/yolov5:latest bash
# Convert onnx to engine
>>>trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine
# Install pycuda
>>>pip3 install pycuda -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

Usage: Infer Image/Video using YOLOv5 with TensorRT and Pytorch:
    $ python3 py/yolov8/yolov8_trt_w_torch.py yolov8n.engine assets/bus.jpg
    $ python3 py/yolov8/yolov8_trt_w_torch.py yolov8n.engine assets/bus.jpg  --video

Usage: Save Image/Video:
    $ python3 py/yolov8/yolov8_trt_w_torch.py yolov8n.engine assets/bus.jpg --save
    $ python3 py/yolov8/yolov8_trt_w_torch.py yolov8n.engine assets/vtest.avi --video --save

"""

import os

import torch
from torch import Tensor

import numpy as np
from numpy import ndarray

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from general import LOGGER
from yolov8_base import YOLOv8Base, pre_transform
from torch_util import check_imgsz, non_max_suppression, scale_boxes, convert_torch2numpy_batch


class YOLOv8TRT(YOLOv8Base):

    def __init__(self, weight: str = 'yolov8n.engine'):
        super().__init__()
        self.load_engine(weight)

        self.device = torch.device("cpu")

    def load_engine(self, weight: str):
        assert os.path.isfile(weight), weight
        LOGGER.info(f'Loading {weight} for TensorRT inference...')

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(weight, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate memory
        self.inputs, self.outputs, self.bindings, self.output_shapes = [], [], [], []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            LOGGER.info(f"{binding} {engine.get_binding_shape(binding)}")
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.output_shapes.append(engine.get_binding_shape(binding))
                self.outputs.append({'host': host_mem, 'device': device_mem})

        self.dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding)))
        LOGGER.info(f"Init Done. Work with {self.dtype}")

    def infer(self, im: Tensor):
        im = im.cpu().numpy()  # torch to numpy

        # Copy input image to host buffer
        self.inputs[0]['host'] = np.ravel(im.astype(self.dtype))
        # Transfer input data to the GPU.
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # Transfer input data to the GPU.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        outputs = [out['host'] for out in self.outputs]
        reshaped = []
        for output, shape in zip(outputs, self.output_shapes):
            reshaped.append(output.reshape(shape))

        y = reshaped
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def preprocess(self, im, imgsz, stride=32, pt=False, fp16=False):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(pre_transform(im, imgsz, stride=stride, pt=pt))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def postprocess(self, preds,
                    img,
                    orig_imgs,
                    # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
                    conf=0.25,
                    iou=0.7,  # (float) intersection over union (IoU) threshold for NMS
                    agnostic_nms=False,  # (bool) class-agnostic NMS
                    max_det=300,  # (int) maximum number of detections per image
                    classes=None,
                    # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
                    ):
        # print("****************************************************")
        # print(f"conf= {conf}")
        # print(f"iou= {iou}")
        # print(f"agnostic_nms= {agnostic_nms}")
        # print(f"max_det= {max_det}")
        # print(f"classes= {classes}")
        # print("****************************************************")
        """Post-processes predictions and returns a list of Results objects."""
        preds = non_max_suppression(preds,
                                    conf_thres=conf,
                                    iou_thres=iou,
                                    agnostic=agnostic_nms,
                                    max_det=max_det,
                                    classes=classes)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = convert_torch2numpy_batch(orig_imgs)
        # print('#' * 20)
        # print(preds[0].reshape(-1)[:20])

        # print(f"names= {self.model.names}")
        # results = []
        for i, pred in enumerate(preds):
            # print(f"img_path = {self.batch[0][i]}")
            orig_img = orig_imgs[i]
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            # img_path = self.batch[0][i]
            # results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))

        # print(f"result len: {len(results)}")
        # for result in results:
        #     print(result.boxes)
        # return results
        # return preds

        pred = preds[0]
        boxes = pred[:, :4]
        confs = pred[:, 4:5]
        cls_ids = pred[:, 5:6]
        return boxes, confs, cls_ids

    def detect(self, im0: ndarray):
        # return super().detect(im0)
        # im = preprocess([im0])
        im0s = [im0]
        im = self.preprocess(im0s, self.imgsz)

        preds = self.infer(im)
        # print("*" * 20)
        # print(preds.reshape(-1)[:20])

        boxes, confs, cls_ids = self.postprocess(preds, im, im0s)
        # boxes, confs, cls_ids = postprocess(outputs, im.shape[2:], im0.shape[:2], conf=0.25, iou=0.45)
        return boxes, confs, cls_ids

    def predict_image(self, img_path, output_dir="output/", suffix="yolov8_trt_w_torch", save=False):
        super().predict_image(img_path, output_dir, suffix, save)

    def predict_video(self, video_file, output_dir="output/", suffix="yolov8_trt_w_torch", save=False):
        super().predict_video(video_file, output_dir, suffix, save)


def parse_opt():
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8TRT Infer")
    parser.add_argument("model", metavar="MODEL", type=str, default='yolov8n.engine',
                        help="Path of TensorRT engine")
    parser.add_argument("input", metavar="INPUT", type=str, default="assets/bus.jpg",
                        help="Path of input, default to image")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Use video as input")

    parser.add_argument("--save", action="store_true", default=False,
                        help="Save or not.")

    args = parser.parse_args()
    LOGGER.info(f"args: {args}")

    return args


def main(args):
    model = YOLOv8TRT(args.model)

    if args.video:
        model.predict_video(args.input, save=args.save)
    else:
        model.predict_image(args.input, save=args.save)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
