# -*- coding: utf-8 -*-
"""时间核。

真实数据（50,465 条微信消息）推翻了 exp(-Δ/S)：
    对数正态（先升后降）LR=+2068.9 优于指数；幂律 6/6 不如指数 [SUMMARY §6.3]。

工程公式（PROJECT_SUMMARY「工程落地参数」）：
    z   = (ln Δ - ln S) / sigma
    w   = exp(-0.5 z²) · [可选: / Δ]
    w  *= exp( c_bnd · ln(Δ/flip) · bnd )          段边界项

口径说明（实现前必须想清的一处）： fitted utility u = -0.5 z² - ln Δ 里的
「-ln Δ」是对**密集消息流**候选底率的比例修正（微信里候选=每条消息）。
对稀疏的事实库，候选不是时间上均匀致密的，1/Δ 会把权重整个推向刚写入的
几秒。文档自己的注释「峰值 0.18 h」指的是 lognormal 因子 exp(-0.5 z²)
本身的峰。所以默认 density_correction=False（纯 lognormal，峰在 S）；
要复现微信条件似然口径时再打开。

S 是「渠道级」参数：逐渠道标定一次后冻结，不存在全局默认值
（14 个会话从 65 s 到 54,283 s，约 835 倍 [SUMMARY §6.4]）。
"""
import math

import numpy as np

from .. import config

# 密度修正默认关闭（见模块 docstring 口径说明）
DENSITY_CORRECTION = False


def _clip_age(age):
    """年龄必须夹到 >=1 秒：事件时间戳晚于查询时刻时 exp 会爆炸 [STORE §4.4]，
    Δ=0 会让 ln 发散。"""
    return np.clip(np.asarray(age, dtype=np.float64), 1.0, None)


def lognormal_logw(age, S=None, sigma=None, density_correction=None):
    """对数正态核的 ln w = -0.5 z² [- ln Δ]。log 空间计算，不下溢成噪声。"""
    S = config.KERNEL_S_DEFAULT if S is None else S
    sigma = config.KERNEL_SIGMA if sigma is None else sigma
    dc = DENSITY_CORRECTION if density_correction is None else density_correction
    a = _clip_age(age)
    z = (np.log(a) - math.log(S)) / sigma
    lw = -0.5 * z * z
    if dc:
        lw = lw - np.log(a)
    return lw


def exp_logw(age, S):
    """指数核 ln w = -Δ/S。保留给标定实验（φ/back 回归的口径）。"""
    a = _clip_age(age)
    return -a / float(S)


def logw(age, seg_bnd=None, S=None, sigma=None, kind=None):
    """带段边界项的总 ln w。seg_bnd 为 None 时按无边界处理（乘子=1）。"""
    kind = config.KERNEL_KIND if kind is None else kind
    if kind == "lognormal":
        lw = lognormal_logw(age, S, sigma)
    elif kind == "exp":
        lw = exp_logw(age, S if S is not None else config.EXP_S_DEFAULT)
    else:
        raise ValueError("unknown kernel kind: %r" % (kind,))
    if seg_bnd is None:
        return lw
    a = _clip_age(age)
    flip = config.BND_FLIP_S
    c = config.BND_C
    lw = lw + np.asarray(seg_bnd, dtype=np.float64) * c * np.log(a / flip)
    return lw


def weight(age, seg_bnd=None, S=None, sigma=None, kind=None):
    """检索权重 w（=exp(lnw)）。只用于相对比较，绝对值没有意义。"""
    return np.exp(logw(age, seg_bnd, S, sigma, kind))


def kernel_peak_age(S=None, sigma=None):
    """lognormal 因子达峰的年龄 = S（文档口径「峰值 0.18 h」= 660 s）。
    打开密度修正后，完整权重的峰移到 S·exp(-σ²)。"""
    S = config.KERNEL_S_DEFAULT if S is None else S
    return S
