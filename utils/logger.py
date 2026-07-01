# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/16
@File    : logger.py
@Author  : zj
@Description: 日志工具 — 项目通用，所有模块共享
"""

import logging


def get_logger(name: str = "root") -> logging.Logger:
    """获取或创建 logger

    Args:
        name: logger 名称

    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
