# -*- coding: utf-8 -*-
"""兼容 shim：性格系统已升级为 dh/growth/（成长层，OCEAN 特质 + 阶段）。

旧 Persona 接口（mu/shape/trust/summary_text/snapshot/write_md/md_path）
由成长引擎同名实现，旧导入路径原样可用；新代码请 import dh.growth。
clamp 是旧实现的公开工具函数，dh/persona.py（更旧的 shim）还在导入
它——签名保持一致（审查 P2-3：删了它旧路径一 import 就炸）。"""
from ..growth.persona import Persona   # noqa: F401


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


__all__ = ["Persona", "clamp"]
