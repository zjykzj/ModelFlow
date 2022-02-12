# -*- coding: utf-8 -*-

"""
@date: 2022/2/12 下午2:45
@file: get_mnist_image.py
@author: zj
@description: 
"""

from PIL import Image
from torchvision import datasets


def main():
    dataset = datasets.MNIST('../data', train=False, transform=None)

    image, target = dataset.__getitem__(10)

    print(type(image))
    print(target)

    assert isinstance(image, Image.Image)
    image.save(f'../assets/mnist_{target}.png')


if __name__ == '__main__':
    main()
