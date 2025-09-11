# -*- coding: utf-8 -*-

"""
@Time    : 2025/8/30 15:57
@File    : yolov8_runtime_w_torch.py
@Author  : zj
@Description: 
"""
import copy
from typing import Union, Tuple, Optional, Any, List

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

from core.backends.backend_runtime import BackendRuntime
from core.utils.general import Profile
from core.utils.v8.preprocessor import LetterBox
from samples.yolov8.torch_util import non_max_suppression, scale_boxes
from samples.yolov8_seg.torch_util import process_mask


def pre_transform(im, imgsz: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False):
    """
    Apply preprocessing transformation (e.g., letterbox resizing) to a list of input images before model inference.

    This function ensures all images are resized to the target size while preserving aspect ratio,
    padding with gray values if necessary (letterboxing). It's typically used when inputs are raw images.

    Args:
        im (List[np.ndarray]): List of input images. Each image is in HWC (height, width, channels) format, BGR color space.
                               Shape: [(H, W, 3)] * N, where N is the number of images.
        imgsz (Union[int, Tuple]): Target input size for the model. If int, a square image (imgsz x imgsz) is assumed.
                                   Can be a tuple (h, w) for rectangular input.
        stride (int): Model's stride (e.g., 32 for YOLOv8). Output shape must be divisible by stride.
                      Used to adjust padding in letterbox.
        auto (bool): If True, enables dynamic padding based on image shape. When combined with `same_shapes=True`,
                     disables padding entirely (i.e., uses simple resize).

    Returns:
        (list): A list of preprocessed NumPy arrays (in HWC format), resized and padded as needed.
                Each element has shape (resized_h, resized_w, 3).
    """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(imgsz, auto=same_shapes and auto, stride=stride)
    return [letterbox(image=x) for x in im]


def preprocess(im: ndarray, imgsz: Union[int, Tuple] = 640, stride: int = 32, auto: bool = False,
               fp16: bool = False) -> Any:
    """
      Sourced from https://github.com/ultralytics/ultralytics/blob/25307552100e4c03c8fec7b0f7286b4244018e15/ultralytics/engine/predictor.py#L115

      Preprocess input image(s) for inference with the YOLOv8 model.

      This function handles both NumPy arrays and PyTorch tensors. For NumPy inputs, it performs:
      - Batch dimension expansion (if missing)
      - Letterbox resizing
      - Color space conversion (BGR -> RGB)
      - Channel ordering (HWC -> CHW)
      - Contiguity enforcement
      - Normalization (0-255 -> 0.0-1.0)
      - Data type conversion (uint8 -> float32 or float16)

      For tensor inputs, only normalization and type conversion are applied.

      Args:
          im (torch.Tensor | List[np.ndarray]): Input image(s).
              - If tensor: Expected in BCHW format (batch, channels, height, width), uint8.
              - If list of arrays: List of HWC images (BGR), no batch dim; will be stacked.
          imgsz (Union[int, Tuple]): Target size for resizing (used only if input is list/array).
          stride (int): Downsample stride of the model (used in letterbox padding alignment).
          auto (bool): Whether to enable auto-padding based on image shape (see `LetterBox`).
          fp16 (bool): If True, convert image tensor to half precision (float16). Otherwise, float32.

      Returns:
          (np.ndarray): Preprocessed image as a NumPy array in BCHW format, normalized to [0, 1],
                        with dtype float16 (if fp16=True) or float32.
                        Shape: (N, 3, resized_h, resized_w)
      """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = np.stack(pre_transform(im, imgsz, stride=stride, auto=auto))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im
    im = im.half() if fp16 else im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im.numpy()


import cv2


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (np.ndarray): The masks that are being returned with shape [h, w, num].
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    print(f"top: {top} - bottom: {bottom} - left: {left} - right: {right}")
    masks = masks[top:bottom, left:right]
    print(f"masks shape: {masks.shape}")
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def plot_masks(masks, colors, im_gpu, im0_shape, alpha=0.5, retina_masks=False):
    """
    Plot masks on image.

    Args:
        masks (tensor): Predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
        im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
        alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
        retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
    """
    if len(masks) == 0:
        im = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
    if im_gpu.device != masks.device:
        im_gpu = im_gpu.to(masks.device)
    colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
    colors = colors[:, None, None]  # shape(n,1,1,3)
    masks = masks.unsqueeze(3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

    im_gpu = im_gpu.flip(dims=[0])  # flip channel
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
    im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
    im_mask = im_gpu * 255
    im_mask_np = im_mask.byte().cpu().numpy()
    im = im_mask_np if retina_masks else scale_image(im_mask_np, im0_shape)
    return im


def plot_masks_v2(masks, colors, im_gpu, im0_shape, alpha=0.5, retina_masks=False):
    """
    Plot masks on image.

    Args:
        masks (tensor): Predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
        im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
        alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
        retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
    """
    print(f"masks shape: {masks.shape}")
    masks = masks.permute(1, 2, 0).numpy()
    print(f"masks shape: {masks.shape}")
    masks = scale_image(masks, im0_shape)
    print(f"last masks shape: {masks.shape}")
    masks = torch.from_numpy(masks).to(im_gpu.device)
    masks = masks.permute(2, 0, 1)
    print(f"last 2 masks shape: {masks.shape}")

    if len(masks) == 0:
        im = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
    if im_gpu.device != masks.device:
        im_gpu = im_gpu.to(masks.device)
    colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
    colors = colors[:, None, None]  # shape(n,1,1,3)
    masks = masks.unsqueeze(3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)l
    mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

    im_gpu = im_gpu.flip(dims=[0])  # flip channel
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
    im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
    im_mask = im_gpu * 255
    im_mask_np = im_mask.byte().cpu().numpy()
    # im = im_mask_np if retina_masks else scale_image(im_mask_np, im0_shape)
    # return im
    return im_mask_np


def masks2segments(masks, strategy="largest"):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy).

    Args:
        masks (torch.Tensor): the output of the model, which is a tensor of shape (batch_size, 160, 160)
        strategy (str): 'concat' or 'largest'. Defaults to largest

    Returns:
        segments (List): list of segment masks
    """
    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if c:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped coordinates
    """
    if isinstance(coords, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (torch.Tensor): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


# Pose
skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]


def plot_kpts(idx, im, kpts, colors, shape=(640, 640), radius=None, kpt_line=True, conf_thres=0.25, kpt_color=None):
    """
    Plot keypoints on the image.

    Args:
        kpts (torch.Tensor): Keypoints, shape [17, 3] (x, y, confidence).
        shape (tuple, optional): Image shape (h, w). Defaults to (640, 640).
        radius (int, optional): Keypoint radius. Defaults to 5.
        kpt_line (bool, optional): Draw lines between keypoints. Defaults to True.
        conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
        kpt_color (tuple, optional): Keypoint color (B, G, R). Defaults to None.

    Note:
        - `kpt_line=True` currently only supports human pose plotting.
        - Modifies self.im in-place.
        - If self.pil is True, converts image to numpy array and back to PIL.
    """
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    radius = radius if radius is not None else lw
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim in {2, 3}
    kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting

    kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    color_k = (kpt_color[idx].tolist() if is_pose else colors(idx))
    for i, k in enumerate(kpts):
        # color_k = kpt_color or (kpt_color[i].tolist() if is_pose else colors(i))
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < conf_thres:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < conf_thres or conf2 < conf_thres:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(
                im,
                pos1,
                pos2,
                kpt_color or limb_color[i].tolist(),
                thickness=int(np.ceil(lw / 2)),
                lineType=cv2.LINE_AA,
            )

    return im


def generate_distinct_colors(num_colors):
    """
    生成对比度高的颜色列表。

    参数:
    - num_colors: 需要生成的颜色数量

    返回:
    - 颜色列表，每个颜色为BGR格式的元组
    """
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)  # 在HSV色彩空间中均匀分布色调
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])))
    return colors


def draw_segmentation_contours(im, segments, colors):
    """
    在原图上绘制每个实例的分割轮廓。

    参数:
    - im: 原始图像（numpy数组格式）
    - segments: 包含每个实例分割坐标点的列表，每个item为numpy数组格式，表示一个实例的分割坐标

    返回:
    - 绘制了分割轮廓的图像
    """
    # 创建一个空白的mask，大小与原图相同，单通道
    mask = np.zeros_like(im) if len(im.shape) == 2 else np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)

    # kpt_color = colors.pose_palette
    kpt_color = generate_distinct_colors(len(segments))
    # kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    for i, segment in enumerate(segments):
        # color_k = kpt_color[i].tolist()
        color_k = kpt_color[i]
        print(f"color_k: {color_k}")
        # color = kpt_color[i]

        # 将segment中的坐标点转换成适合fillPoly的格式
        pts = segment.reshape((-1, 1, 2)).astype(np.int32)
        # 使用白色填充轮廓区域，也可以选择其他颜色
        # cv2.fillPoly(mask, [pts], color=color_k)
        # cv2.fillPoly(mask, [pts], color=(255, 255, 255))
        # 如果需要只绘制轮廓线而不是填充区域，可使用以下代码替代fillPoly部分：
        # cv2.polylines(mask, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(mask, [pts], isClosed=True, color=color_k, thickness=2)

    # 将mask与原图结合
    # result = cv2.addWeighted(im, 1.0, mask, 0.4, 0)
    result = cv2.addWeighted(im, 1.0, mask, 1.0, 0)

    return result


def postprocess(
        im0,
        pred: Union[Tensor, List[Tensor]],
        im_shape: Tuple,  # (h, w) of input to model
        im0_shape: Tuple,  # (h, w) of original image
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[list] = None,
        agnostic: bool = False,
        max_det: int = 300,
        nc: int = 0,  # number of classes (optional)
) -> Tuple:
    """
    Post-process model predictions (detections) after inference.

    Applies Non-Max Suppression (NMS) to filter overlapping bounding boxes,
    scales detection boxes back to original image coordinates, and separates outputs.

    Args:
        pred (Union[Tensor, List[Tensor]]): Raw model output detections. Shape: (batch, num_boxes, 4 + 1 + num_classes)
        im_shape (Tuple): Shape of the image fed into the model (after preprocessing), as (height, width).
        im0_shape (Tuple): Original shape of the input image (before any preprocessing), as (height, width).
        conf (float): Confidence threshold for filtering detections.
        iou (float): IoU threshold for NMS.
        classes (Optional[list]): List of class indices to keep. If None, keep all classes.
        agnostic (bool): If True, perform NMS across all classes (class-agnostic).
        max_det (int): Maximum number of detections to keep per image.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.

    Returns:
        (Tuple): A tuple containing:
            - boxes (np.ndarray or []): Scaled bounding boxes in xyxy format, shape (N, 4), relative to original image.
            - confs (np.ndarray or []): Confidence scores for each kept detection, shape (N, 1).
            - cls_ids (np.ndarray or []): Predicted class IDs, shape (N, 1).
            If no detections, returns empty lists.
    """
    proto = pred[1][-1] if isinstance(pred[1], tuple) else pred[1]  # tuple if PyTorch model or array if exported
    proto = proto[0]  # [1, 32, 160, 160] -> [32, 160, 160]

    pred = non_max_suppression(
        pred[0],
        conf,
        iou,
        classes=classes,
        agnostic=agnostic,
        max_det=max_det,
        nc=nc,
    )
    pred = pred[0]  # [1, 300, 6] -> [300, 6]

    masks = process_mask(proto, pred[:, 6:], pred[:, :4], im_shape, upsample=True)  # HWC

    # img = LetterBox(im_shape)(image=im0)
    # im_gpu = (
    #         torch.as_tensor(img, dtype=torch.float16, device=torch.device("cpu"))
    #         .permute(2, 0, 1)
    #         .flip(0)
    #         .contiguous()
    #         / 255
    # )
    #
    idx = reversed(range(len(masks)))
    from annotator import colors
    # im = plot_masks(masks, colors=[colors(x, True) for x in idx], im_gpu=im_gpu, im0_shape=im0_shape)
    # print(f"im shape: {im.shape} - im type: {type(im)}")
    # cv2.imwrite("im_seg.jpg", im)

    im_gpu = (
            torch.as_tensor(im0, dtype=torch.float16, device=torch.device("cpu"))
            .permute(2, 0, 1)
            .flip(0)
            .contiguous()
            / 255
    )
    im = plot_masks_v2(masks, colors=[colors(x, True) for x in idx], im_gpu=im_gpu, im0_shape=im0_shape)
    cv2.imwrite("im_masks.png", im)

    # segments = [scale_coords(im_shape, x, im0_shape, normalize=False) for x in masks2segments(masks)]
    # print(f"segments len: {len(segments)}")
    # for item in segments:
    #     print(f"item shape: {item.shape}")
    # print(segments[-1])
    #
    # im = copy.deepcopy(im0)
    # for idx, segment in enumerate(segments):
    #     im = plot_kpts(idx, im, segment, colors)
    # cv2.imwrite("im_pts.jpg", im)
    #
    # im2 = copy.deepcopy(im0)
    # im2 = draw_segmentation_contours(im2, segments, colors)
    # cv2.imwrite("im2_pts.jpg", im2)

    if len(pred) > 0:
        boxes = scale_boxes(im_shape, pred[:, :4], im0_shape)
        confs = pred[:, 4:5]
        cls_ids = pred[:, 5:6]
    else:
        boxes, confs, cls_ids = [], [], []
    return boxes, confs, cls_ids


class YOLOv8RuntimeTorch:

    def __init__(self, classes: List[str], weight: str = 'yolov8s.onnx', providers: List[str] = None):
        super().__init__()
        self.classes = classes
        self.nc = len(classes)

        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = BackendRuntime(weight, providers=providers)
        self.session.load()

        self.input_name = self.session.get_input_names()[0]
        self.net_h, self.net_w = self.session.get_input_shapes()[self.input_name][2:]
        self.output_names = self.session.output_names

        self.device = torch.device('cpu')

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def infer(self, im: ndarray) -> List[Any]:
        output_dict = self.session.infer({self.input_name: im})

        pred = []
        for output_name in self.output_names:
            pred.append(self.from_numpy(output_dict[output_name]))
        return pred

    def detect(self, im0: ndarray, conf: float = 0.25, iou: float = 0.45) -> Tuple:
        """
        Detect objects in the image and measure time consumption for each stage.
        Returns:
            boxes, confs, cls_ids, dt
        """
        # Record start time
        dt = (Profile(), Profile(), Profile())

        # --- Preprocessing ---
        with dt[0]:
            im = preprocess(im0, (self.net_h, self.net_w))
            im_shape = im.shape[2:]  # Model input shape (h, w)
            im0_shape = im0.shape[:2]  # Original image shape (h, w)

        # --- Inference ---
        with dt[1]:
            pred = self.infer(im)

        # --- Postprocessing ---
        with dt[2]:
            boxes, confs, cls_ids = postprocess(
                im0,
                pred,
                im_shape,
                im0_shape,
                conf=conf,
                iou=iou,
                nc=self.nc
            )

        return boxes, confs, cls_ids, dt
