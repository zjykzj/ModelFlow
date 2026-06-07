# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : logger.py
@Author  : zj
@Description: 日志工具
"""

import logging
from typing import Optional

LOGGER_NAME = "modelflow"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取或创建 logger

    Args:
        name: logger 名称（默认 "modelflow"）

    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name or LOGGER_NAME)
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
