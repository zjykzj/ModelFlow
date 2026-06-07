# -*- coding: utf-8 -*-
"""
@Time    : 2026/6/7
@File    : registry.py
@Author  : zj
@Description: 组件注册机制

支持装饰器注册和按名构建，新增后端/任务/数据集不改核心框架代码。

用法:
    @BACKENDS.register("my_backend")
    class MyBackend(BaseBackend): ...

    backend = BACKENDS.build("my_backend", model_path="...")
"""

from typing import Callable, Dict, Any, List, Optional, Type


class Registry:
    """组件注册器

    Args:
        name: 注册器名称（用于日志/调试）
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """装饰器：注册一个类

        Args:
            name: 注册名（默认使用类名小写）

        Returns:
            装饰器函数
        """
        def wrapper(cls: Type) -> Type:
            key = name or cls.__name__.lower()
            if key in self._registry:
                print(f"[Registry] ⚠️  Overwriting {self._name}[{key!r}]")
            self._registry[key] = cls
            return cls
        return wrapper

    def get(self, name: str) -> Type:
        """按名获取已注册的类

        Args:
            name: 注册名

        Returns:
            类对象
        """
        if name not in self._registry:
            raise KeyError(f"[Registry] {self._name}[{name!r}] not registered. "
                           f"Available: {self.list()}")
        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        """按名构建组件实例

        Args:
            name: 注册名
            **kwargs: 传给构造函数的参数

        Returns:
            组件实例
        """
        cls = self.get(name)
        return cls(**kwargs)

    def list(self) -> List[str]:
        """列出所有已注册的组件名"""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, {len(self._registry)} items)"


# 全局注册器
BACKENDS = Registry("backends")
PROCESSORS = Registry("processors")
DATASETS = Registry("datasets")
METRICS = Registry("metrics")
EVALUATORS = Registry("evaluators")
