# -*- coding: utf-8 -*-
"""兼容 shim：情绪系统已迁至 dh/limbic/emotion.py（右脑，见 dh/limbic/__init__）。

旧导入路径原样可用（历史引用方均已迁往 dh.limbic，此处只为外部兼容保留）；
新代码请直接 import dh.limbic.emotion。"""
from ..limbic.emotion import (          # noqa: F401
    PAD, PAD_DIM, Appraisal, appraise_to_pad,
    OUDynamics, ReactionBuffer, AffectSystem,
    outlet_language, outlet_speech, outlet_face, outlet_memory_gate,
    autocov, estimate_theta_c2c1, estimate_theta_raw,
)

__all__ = [
    "PAD", "PAD_DIM", "Appraisal", "appraise_to_pad",
    "OUDynamics", "ReactionBuffer", "AffectSystem",
    "outlet_language", "outlet_speech", "outlet_face", "outlet_memory_gate",
    "autocov", "estimate_theta_c2c1", "estimate_theta_raw",
]
