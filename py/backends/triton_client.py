# -*- coding: utf-8 -*-

"""
@Time    : 2024/10/13 14:29
@File    : triton_client.py
@Author  : zj
@Description: 
"""

import threading
from tritonclient import grpc as grpcclient


class TritonClientFactory:
    _lock = threading.Lock()  # 创建一个锁
    _client = None  # 存储客户端实例

    @classmethod
    def get_client(cls, url="localhost:8001"):
        """获取或创建 Triton 客户端"""
        with cls._lock:  # 使用上下文管理器来自动处理锁的获取和释放
            if cls._client is None:
                cls._client = grpcclient.InferenceServerClient(url)
                print("Create Triton Client, binding to", url)
        return cls._client
