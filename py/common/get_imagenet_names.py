# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 上午9:59
@file: get_imagenet_names.py
@author: zj
@description: 
"""

import json

import numpy as np

if __name__ == '__main__':
    imagenet_path = '../../assets/imagenet_class_index.json'
    with open(imagenet_path, 'r') as f:
        data_dict = json.load(f)

    print(data_dict)
    print(np.array(list(data_dict.keys())).astype(int))

    cls_names = list()
    for key in sorted(np.array(list(data_dict.keys())).astype(int)):
        print(key, data_dict[str(key)])
        cls_names.append(data_dict[str(key)][1])

    print(cls_names)
    np.savetxt('../../assets/imagenet.names', cls_names, fmt='%s', delimiter=' ')
    print('done')
