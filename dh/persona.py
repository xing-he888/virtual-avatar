# -*- coding: utf-8 -*-
"""兼容 shim：性格慢塑形已迁至 dh/limbic/persona.py（右脑，与情绪同族）。

旧导入路径原样可用；新代码请直接 import dh.limbic.persona。"""
from .limbic.persona import Persona, clamp  # noqa: F401

__all__ = ["Persona", "clamp"]
