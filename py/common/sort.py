# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 上午9:42
@file: sort.py
@author: zj
@description: 
"""
import PIL.Image
import torch
import torchvision
from torchvision import transforms
from torchvision.models.resnet import resnet18

if __name__ == '__main__':
    model = resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    print(model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.229, 0.224, 0.225], std=[0.485, 0.456, 0.406])
    ])

    image = PIL.Image.open("../../assets/ILSVRC2012_val_00010244.JPEG")
    data = transform(image)

    res = model(data.unsqueeze(0))[0]
    print(res)

    sort_values, sort_indices = torch.sort(res, descending=True)
    print(sort_values)
    print(sort_indices)
