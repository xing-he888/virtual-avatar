# -*- coding: utf-8 -*-
"""混合打分：z 标准化 + 名义权重 + 两条断言。

三条实现纪律 [AUDIT §8]，每一条都来自实测翻车现场：
    1. 每项先在本查询的候选集内 z 标准化，否则名义权重是假的
       （名义 0.3 的情绪项实际只占 0.229，最差时情绪项只有相关项的 37.4%）。
    2. 两条断言同时写：方差占比（抓尺度不一致）+ 分布形状（抓「抽样没给 size」
       这类把整列压扁的 bug —— sal 标量 bug 的 sd 是 0.05，正确是 0.17）。
    3. 门槛 K 与权重分开调。结构比权重重要一个量级（0.867 vs ~0.1）。
"""
import numpy as np

from . import config


def zstd(x, mask=None):
    """本查询候选集内的 z 标准化。全常数时返回全 0（该项在本查询无信息）。"""
    x = np.asarray(x, dtype=np.float64)
    m = np.ones(len(x), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.sum() < 2:
        return np.zeros_like(x)
    mu = x[m].mean()
    sd = x[m].std()
    if sd < 1e-12:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[m] = (x[m] - mu) / sd
    return out


def check_weights(terms, w, mask=None):
    """断言 1：名义权重 vs 实际方差占比 [AUDIT §3]。

    必须拿「真正喂进加法的那几项」测（z 标准化后的输入永远 PASS）。
    返回 (ok, dev, share)。
    """
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    m = np.ones(len(terms[0]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    sd = np.array([np.asarray(t, dtype=np.float64)[m].std() for t in terms])
    c = w * sd
    share = c / c.sum()
    dev = float(np.abs(share - w).max())
    return dev <= config.SCORE_RTOL, dev, share


def check_dist(x, lo, hi):
    """断言 2：分布形状，专门抓「抽样没给 size」把整列压扁的 bug。"""
    s = float(np.asarray(x, dtype=np.float64).std())
    return (lo <= s <= hi), s


def hybrid_score(sim, time_w, emo, w=None, seg_mask=None):
    """三路召回的混合分：z 标准化后按名义权重相加 [DESIGN §3.2 三路]。

    sim   语义路（余弦）
    time_w 时间路（时间核权重，已经 >=0）
    emo   情绪路（情绪一致度，1 + bias·mood·valence）
    返回 (score, terms_z)。terms_z 供断言用。
    """
    w = w or config.HYBRID_W
    wv = np.array([w["sim"], w["time"], w["emo"]], dtype=np.float64)
    terms = [np.asarray(sim, dtype=np.float64),
             np.asarray(time_w, dtype=np.float64),
             np.asarray(emo, dtype=np.float64)]
    tz = [zstd(t, seg_mask) for t in terms]
    score = wv[0] * tz[0] + wv[1] * tz[1] + wv[2] * tz[2]
    return score, tz
