# -*- coding: utf-8 -*-
"""情绪韵律的音频后处理：把情绪四出口里的 pitch/energy 落到波形上。

四出口（dh/emotion.py outlet_speech）的分工：
    rate   -> CosyVoice 的 speed 参数（模型原生支持，不在这里做）
    pitch  -> 音高移位（半音数 = 12*log2(倍数)，librosa）
    energy -> 振幅增益（峰值超限整体压回，防爆音）

纯函数、无模型依赖——测试直接喂合成正弦波。
"""
import numpy as np

PEAK = 0.98          # 合成音频的峰值上限，超过就算爆音


def semitones(factor):
    """倍频系数 -> 半音数（pitch=1.5 即上移纯五度）。"""
    return 12.0 * float(np.log2(factor))


def apply_pitch(audio, sr, factor=1.0):
    if abs(factor - 1.0) < 1e-3:
        return audio
    import librosa                      # 重依赖，按需加载
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones(factor))


def apply_energy(audio, factor=1.0):
    if abs(factor - 1.0) < 1e-3:
        return audio
    out = audio * float(factor)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > PEAK:                     # 增益把波形顶过上限 -> 等比压回
        out = out * (PEAK / peak)
    return out


def apply_prosody(audio, sr, pitch=1.0, energy=1.0):
    return apply_energy(apply_pitch(audio, sr, pitch), energy)
