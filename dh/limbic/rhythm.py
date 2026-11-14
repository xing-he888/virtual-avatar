# -*- coding: utf-8 -*-
"""昼夜节律：情绪基线的时刻调制（时间动态更新的半边）。

三层分工，各管一段：
    成长层（dh/growth/）  定「这个人平时什么底色」——慢变量，被经历塑形
    本模块               定「这一天这个时刻底色偏向哪」——24h 日周期
    OU 动力学            时间让情绪「回到」基线；本模块让基线「自己走」

模型：每维一个余弦。不用单调性、但保持全项目的纪律——有界、可在极值处
查表验证的初等函数，不用学习参数：

    offset_i(h) = amp_i · cos(2π (h − peak_i) / 24)

    极值表：h = peak   → +amp（峰）
            h = peak+12 → −amp（谷）
            任意 h     → |offset| ≤ amp
            周期性：offset(h+24) = offset(h)

三维的峰谷方向取自昼夜节律实证的定性结论（出处见 config.EMO_DIURNAL_*）：
正性情感/清醒度谷在凌晨、峰在午后到傍晚；支配维跟清醒度同向但波动最小
——凌晨的「底气不足」比「心情差」更接近现象学。

时刻从本地钟面取（hour_of）：小星的作息跟着用户的时区走，不跟 UTC。
"""
import time

import numpy as np

from .. import config


def hour_of(ts=None):
    """钟面小时（浮点，含分秒）。AffectSystem 默认从这里取时刻。

    ts 缺省用当前墙钟；测试注入固定 epoch 即得固定小时（同种子同结果）。
    """
    lt = time.localtime(time.time() if ts is None else ts)
    return lt.tm_hour + lt.tm_min / 60.0 + lt.tm_sec / 3600.0


def offset(hour, amp=None, peak=None):
    """24h 周期余弦偏移，PAD 三维。hour: 钟面小时（浮点，可向量）。"""
    amp = config.EMO_DIURNAL_AMP if amp is None else np.asarray(amp)
    peak = config.EMO_DIURNAL_PEAK if peak is None else np.asarray(peak)
    h = np.asarray(hour, dtype=np.float64)
    return amp * np.cos(2.0 * np.pi * (h - peak) / 24.0)


def on():
    """节律开关（config.EMO_DIURNAL_ON）。关掉时基线退回纯性格。"""
    return bool(config.EMO_DIURNAL_ON)
