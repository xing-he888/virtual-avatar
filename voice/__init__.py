# -*- coding: utf-8 -*-
"""voice 包：语音合成与韵律后处理。

TTS 平台分派（get_tts_class）：
    Windows -> voice/tts.py（用户既有的 CosyVoice 布局，原样保留，不做任何改动）
    其它平台 -> voice/tts_mac.py（项目内 third_party/ 布局，零硬编码路径，
                设备自适应 cuda -> mps -> cpu，全部可用环境变量覆盖）
两个文件接口完全一致（speak / play / stop），调用方不感知平台差异。
"""
import os


def get_tts_class():
    if os.name == "nt":
        from .tts import TTS
    else:
        from .tts_mac import TTS
    return TTS


def tts_model_ready():
    """轻量探测模型是否就位（不触发 torch/cosyvoice 重导入）。

    Windows 版 voice/tts.py 的模型路径只有加载时才知道，这里乐观返回 True，
    真实可用性由首次点 🔊 的加载结果裁决（失败会经 /api/state 报出来）。
    """
    if os.name == "nt":
        return True
    from . import tts_mac
    return tts_mac.model_ready()
