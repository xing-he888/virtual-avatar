# -*- coding: utf-8 -*-
"""遗忘常数的标定：从系统自己的观测流里估，估完冻结。

三条实测依据 [SUMMARY 正面结论 1/2/3、DESIGN §3.1/§3.3]：
    1. 量级压倒形状：常数放错量级，效用 +0.28~0.53；放对 +0.76。
       75% 的可改进空间卡在「量级对不对」上。
    2. 两个不花钱的统计量就能估：
           phi  = 相邻两次观测值不相等的比例（翻转率）
           back = A→B→A 的比例（回翻率；真实世界不会立刻翻回，回翻≈识别出错）
       ln S = 4.5345 - 0.2477·ln(phi) + 0.7246·ln(back)    (训练 R²=0.8186)
    3. 特征不够时换形状会失灵；游程直方图（F1，5 项）在四种留出形状上
       回收 0.992-0.994。

冻结纪律 [DESIGN §3.3]：会话开始标定一次，然后锁死。
在线自适应实测输给写死的常数（+0.7877 vs +0.8064）。
换领域必须在新领域的流上重标定一次再冻结 [DESIGN §3.4]。

口径警告：上面的回归是**指数核**口径（research/exp_v16）。
运行时默认核是对数正态，两种 S 差一个形状换算，所以这里把「标定值」
原样存进 calib 表（标的是哪种核也存），检索层按核类型取用。
"""
import math
from collections import Counter, OrderedDict

import numpy as np

from .. import config


# ---------------------------------------------------------------- 特征统计
class StreamStats:
    """一个观测流（同一实体谓词按时间排好的值序列）的两个计数器 + F1 特征。

    内存 O(1)：只留上一个值、上上个值和几个计数器；F1 的游程也只留当前段。
    """

    def __init__(self):
        self.n = 0
        self.n_flip = 0          # 相邻不等的次数
        self.n_back = 0          # A→B→A 的次数（三连）
        self.n_backable = 0      # 有机会回翻的三连数
        self._prev = None
        self._prev2 = None
        # 游程直方图（F1）：当前段长度与历史段的长度分布
        self._run = 0
        self._run_hist = Counter()
        self._vals = Counter()   # 值域分布（算熵用；键数=值域大小）

    def observe(self, value):
        self.n += 1
        self._vals[value] += 1
        if self.n >= 2 and self._prev != value:
            self.n_flip += 1
        if self.n >= 3:
            self.n_backable += 1
            if self._prev != value and self._prev2 == value:
                self.n_back += 1
        # 游程
        if self._prev is not None and value != self._prev:
            self._run_hist[self._run] += 1
            self._run = 1
        else:
            self._run += 1
        self._prev2, self._prev = self._prev, value

    @property
    def phi(self):
        """翻转率：相邻两次观测值不相等的比例。"""
        if self.n < 2:
            return None
        return self.n_flip / (self.n - 1)

    @property
    def back(self):
        """回翻率：A→B→A 占所有三连的比例。"""
        if self.n_backable == 0:
            return None
        return self.n_back / self.n_backable

    def f1_features(self):
        """游程直方图 F1，5 项 [SUMMARY 正面结论 3]：
        连续不变段长度=1/2/>=3 的比例、平均游程、值域熵。"""
        if self.n == 0:
            return None
        hist = dict(self._run_hist)
        hist[self._run] = hist.get(self._run, 0) + 1     # 收尾的当前段
        total = sum(hist.values())
        r1 = hist.get(1, 0) / total
        r2 = hist.get(2, 0) / total
        r3p = sum(v for k, v in hist.items() if k >= 3) / total
        mean_run = sum(k * v for k, v in hist.items()) / total
        vals = np.array(list(self._vals.values()), dtype=np.float64)
        p = vals / vals.sum()
        ent = float(-(p * np.log(p)).sum())              # 值域熵（自然对数）
        return dict(run1=r1, run2=r2, run3p=r3p, mean_run=mean_run, ent=ent)


# ---------------------------------------------------------------- 回归
def regression_lnS(phi, back):
    """ln S = 4.5345 - 0.2477·ln(phi) + 0.7246·ln(back)。

    phi/back 任一为 0（世界从不翻 / 从不回翻）时回归失效：
    phi=0 说明世界几乎不变，外推会给出荒谬的大 S——这是 E49-E52 实测过的
    外推失败（对话世界 phi=0.0345 落在训练分布最左下角，公式给 8.5、真扫 512，
    差 60 倍）。所以这里夹住输入并把警告还给调用方。
    """
    if phi is None or back is None or phi <= 0.0 or back <= 0.0:
        return None
    a, b, c = config.CALIB_REG
    phi = min(max(phi, 1e-4), 1.0)
    back = min(max(back, 1e-4), 1.0)
    return a + b * math.log(phi) + c * math.log(back)


def estimate_S_exp(stats):
    """从 StreamStats 估指数核口径的 S（秒）。估不出返回 None。"""
    ln_s = regression_lnS(stats.phi, stats.back)
    return None if ln_s is None else math.exp(ln_s)


# ---------------------------------------------------------------- 冻结调度
class CalibSchedule:
    """启动时间表 [SUMMARY 启动时间表]：

        观测条数     用哪套特征
        0-50        渠道默认值（新渠道没有全局默认，用配置里的渠道先验）
        50-150      phi + back 两计数回归（指数口径）
        >=150       游程直方图特征一并记录
        然后        冻结，不再改

    「冻结」落在 calib 表上就是：只插新行，不 update [STORE §6]。
    """

    def __init__(self, store, scope="session", key="kernel_S"):
        self.store = store
        self.scope = scope
        self.key = key
        self.frozen = False

    def observe(self, stats):
        """喂一次观测流统计，返回 (S, 特征集名, 是否新写入)。"""
        if self.frozen:
            return self.store.calib_latest(self.scope, self.key, "S"), "frozen", False
        if stats.n < config.CALIB_N_SMALL:
            return None, "default", False
        s_est = estimate_S_exp(stats)
        feats = "f0" if stats.n < config.CALIB_N_FULL else "f1"
        if s_est is not None:
            self.store.calib_set(self.scope, self.key, "S", s_est, note=feats)
        return s_est, feats, True

    def freeze(self):
        self.frozen = True
