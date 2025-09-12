# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/11 19:49
@File    : ops.py
@Author  : zj
@Description: 
"""

import cv2
import scipy.ndimage

import numpy as np


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (numpy.ndarray): Clipped coordinates
    """
    # np.array (faster grouped)
    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (np.ndarray): the coords to be scaled of shape (n, 2).
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (np.ndarray): The scaled coordinates of shape (n, 2).
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


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    # if isinstance(boxes, torch.Tensor):  # faster individually
    #     boxes[..., 0].clamp_(0, shape[1])  # x1
    #     boxes[..., 1].clamp_(0, shape[0])  # y1
    #     boxes[..., 2].clamp_(0, shape[1])  # x2
    #     boxes[..., 3].clamp_(0, shape[0])  # y2
    # else:  # np.array (faster grouped)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def masks2segments(masks, strategy="largest"):
    """
    Convert batched binary masks to segmentation contours.

    Args:
        masks (np.ndarray): Binary masks array of shape (N, H, W), values 0 or 1.
        strategy (str): 'concat' to merge all contours; 'largest' to keep the biggest one.

    Returns:
        List[np.ndarray]: List of float32 arrays, each of shape (K, 2), representing [x, y] points.
                         Empty list element is (0, 2) if no contour found.
    """
    segments = []
    for mask in masks.astype("uint8"):
        # Handle OpenCV version difference
        cnt_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnt_info[0] if len(cnt_info) == 2 else cnt_info[1]

        if contours:
            if strategy == "concat":
                points = [contour.reshape(-1, 2) for contour in contours]
                c = np.concatenate(points, axis=0)
            elif strategy == "largest":
                sizes = [len(contour) for contour in contours]
                c = contours[np.argmax(sizes)].reshape(-1, 2)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            c = np.zeros((0, 2), dtype=np.float32)

        segments.append(c.astype(np.float32))
    return segments


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Crop masks to their corresponding bounding boxes.

    Args:
        masks (np.ndarray): Array of shape [N, H, W], where N is the number of masks.
        boxes (np.ndarray): Array of shape [N, 4], in format [x1, y1, x2, y2] (absolute coordinates).

    Returns:
        (np.ndarray): Cropped masks, same shape as input masks. Values outside the boxes are set to 0.
    """
    n, h, w = masks.shape

    # Expand boxes to shape [N, 4, 1, 1]
    boxes_expanded = boxes[:, :, np.newaxis, np.newaxis]  # shape: (n, 4, 1, 1)
    x1, y1, x2, y2 = boxes_expanded[:, 0], boxes_expanded[:, 1], boxes_expanded[:, 2], boxes_expanded[:, 3]
    # Each of x1, y1, x2, y2 has shape: (n, 1, 1)

    # Create coordinate grids: r (columns) for x, c (rows) for y
    r = np.arange(w, dtype=x1.dtype)[np.newaxis, np.newaxis, :]  # shape: (1, 1, w)
    c = np.arange(h, dtype=y1.dtype)[np.newaxis, :, np.newaxis]  # shape: (1, h, 1)

    # Create binary mask for inside-box region using broadcasting
    # Shape of condition: (n, h, w)
    inside_box = (r >= x1) & (r < x2) & (c >= y1) & (c < y2)  # bool array, broadcasted

    # Apply mask
    return masks * inside_box  # zero out pixels outside the box


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (np.ndarray): A ndarray of shape [mask_dim, mask_h, mask_w].
        masks_in (np.ndarray): A ndarray of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (np.ndarray): A ndarray of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (np.ndarray): A binary mask ndarray of shape [n, h, w], where n is the number of masks after NMS,
                      and h and w are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape

    # 矩阵乘法与reshape
    masks = np.dot(masks_in, protos.reshape(c, -1)).reshape(-1, mh, mw)  # CHW

    # 计算比例因子
    width_ratio = mw / iw
    height_ratio = mh / ih

    # 缩放边界框坐标
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 1] *= height_ratio
    downsampled_bboxes[:, 3] *= height_ratio

    # 裁剪掩码
    masks = crop_mask(masks, downsampled_bboxes)  # CHW

    # 上采样到原始图像尺寸
    if upsample:
        masks = np.array([scipy.ndimage.zoom(mask, (ih / mh, iw / mw), order=1) for mask in masks])

    # 转换为二值掩码
    return np.greater(masks, 0.0).astype(np.float32)


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
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks
