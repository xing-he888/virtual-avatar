# -*- coding: utf-8 -*-
"""兼容 shim：情绪→行为倾向的调制面已迁至 dh/limbic/disposition.py（右脑）。

旧导入路径原样可用；新代码请直接 import dh.limbic.disposition。"""
from ..limbic.disposition import Disposition, disposition, gate  # noqa: F401

__all__ = ["Disposition", "disposition", "gate"]
