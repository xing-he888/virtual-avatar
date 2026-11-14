# -*- coding: utf-8 -*-
"""情绪子系统：PAD 连续状态 + OU 动力学 + 四个出口 + C2/C1 估计量。

    表征   PAD 三维连续向量（离散标签只在输出时生成）[DESIGN §4.1]
    生成   OCC/顺序评价 -> PAD 增量（系数显式可调，不藏在网络里）
    动力学 OU 精确离散化；衰减速度是人格参数（Davidson 2000）[DESIGN §4.2]
    出口   语言读 P+D / 语音读 A / 表情读 A+P / 记忆写入门读 A [DESIGN §4.3]
    缓冲   五门心路 17 心刹那 -> 最小反应窗口 + 余晖 [DESIGN §1]

估计量纪律 [SUMMARY 正面结论 4]：
    拿带噪声的读数直接拟 AR(1)（a=C1/C0）会把 theta 高估
    （信度 0.5 时 +239%）。用 a = C2/C1（观测噪声只污染 k=0 格）偏差 <=4%。
"""
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .. import config
from . import rhythm

PAD_DIM = 3


@dataclass
class PAD:
    p: float = 0.0   # 效价 [-1,1]
    a: float = 0.0   # 唤醒 [-1,1]
    d: float = 0.0   # 支配 [-1,1]

    def as_array(self):
        return np.array([self.p, self.a, self.d], dtype=np.float64)

    @staticmethod
    def from_array(v):
        return PAD(float(v[0]), float(v[1]), float(v[2]))

    def clipped(self):
        return PAD(*(float(np.clip(x, -1.0, 1.0)) for x in self.as_array()))

    def __add__(self, o):
        return PAD(self.p + o.p, self.a + o.a, self.d + o.d)

    def __mul__(self, k):
        return PAD(self.p * k, self.a * k, self.d * k)


@dataclass
class Appraisal:
    """顺序评价的输入。缺省值表示「未评价」，与 0 区分。"""
    desirability: float = 0.0
    likelihood: float = 1.0
    agency: str = "circumstance"     # self / other / circumstance
    novelty: float = 0.0
    goal_relevance: float = 1.0
    controllability: float = 0.5
    norm_compliance: float = 0.0
    certainty: float = 0.5


def appraise_to_pad(ap: Appraisal) -> PAD:
    """Scherer 顺序检查 + OCC 组合 -> PAD 增量。系数显式可调。"""
    g = float(np.clip(ap.goal_relevance, 0.0, 1.0))
    if g <= 1e-6:
        return PAD()
    expect = ap.desirability * ap.likelihood
    p = expect + 0.35 * ap.norm_compliance
    a = 0.55 * ap.novelty + 0.45 * abs(ap.desirability) * ap.likelihood
    d = 1.6 * (ap.controllability - 0.5)
    if ap.agency == "self":
        d += 0.35
    elif ap.agency == "other":
        d -= 0.35
    d += 0.25 * (ap.certainty - 0.5)
    return (PAD(p, a, d) * g).clipped()


@dataclass
class OUDynamics:
    """每维独立 OU，精确离散化：
        s_{t+1} = mu(1-a) + a·s_t + eps,  a = exp(-theta·dt)
    theta 小 -> 惯性大 -> 情绪粘住；sigma 大 -> 无刺激也不安静。
    """
    theta: np.ndarray = field(default_factory=lambda: config.EMO_THETA.copy())
    sigma: np.ndarray = field(default_factory=lambda: config.EMO_SIGMA.copy())
    mu: np.ndarray = field(default_factory=lambda: config.EMO_MU.copy())

    def step(self, s: PAD, dt_min: float, drive: Optional[PAD] = None,
             rng: Optional[np.random.Generator] = None) -> PAD:
        rng = rng or np.random.default_rng()
        v = s.as_array()
        th, sg = self.theta, self.sigma
        a = np.exp(-th * dt_min)
        mean = self.mu * (1 - a) + a * v
        if drive is not None:
            # 外部驱动注入。权重取 (1-a) 但设下限：tick 粒度再细，
            # 一次评价事件也按至少 EMO_DRIVE_MIN 的强度进状态
            w_drive = np.maximum(1.0 - a, config.EMO_DRIVE_MIN)
            mean = mean + drive.as_array() * w_drive
        sd = np.sqrt(sg ** 2 * (1 - a ** 2) / (2 * np.maximum(th, 1e-9)))
        return PAD.from_array(mean + sd * rng.standard_normal(PAD_DIM)).clipped()

    def mean_revert(self, s: PAD, dt_min: float) -> PAD:
        """无噪声的均值回归：s -> mu + (s-mu)·exp(-theta·dt)。

        长间隔回归的原语（decay 的长间隔分支调它，隔夜/隔天）：OU 的大步长
        端点是「从平稳分布重新抽签」（平稳 sd≈0.56），一天的随机波动不该靠
        一次抽签——那部分归成长层塑形管；这里只做确定性均值回归，即「忘掉」。
        mu 由调用方在步进前 refresh 到「终点时刻」（见 decay 长间隔分支注释）。"""
        v = s.as_array()
        a = np.exp(-self.theta * dt_min)
        mean = self.mu * (1 - a) + a * v
        return PAD.from_array(mean).clipped()

    def half_life(self):
        return np.log(2.0) / np.maximum(self.theta, 1e-9)


@dataclass
class ReactionBuffer:
    """速行 ×7 = 反应有固定最小窗口，不可中断（防抖）；彼所缘 = 有限余晖。"""
    refractory_ms: float = 250.0
    afterglow_ms: float = 900.0
    _last_ignite_ms: Optional[float] = None
    _pending: List[PAD] = field(default_factory=list)

    def can_ignite(self, now_ms):
        return self._last_ignite_ms is None or (now_ms - self._last_ignite_ms) >= self.refractory_ms

    def offer(self, now_ms, delta: PAD):
        if self.can_ignite(now_ms):
            self._last_ignite_ms = now_ms
            return True
        self._pending.append(delta)
        return False

    def drain(self, now_ms) -> PAD:
        if not self._pending:
            return PAD()
        acc = PAD()
        for i, d in enumerate(self._pending):
            acc = acc + d * math.exp(-i / max(1.0, self.afterglow_ms / 100.0))
        self._pending.clear()
        return acc


# ---------------------------------------------------------------- 四出口
def outlet_language(pad: PAD):
    """语言通道读 P 与 D。A 不进语言，避免「喊叫式文字」。"""
    return {"warmth": pad.p, "assertiveness": pad.d,
            "verbosity": float(np.clip(0.5 + 0.5 * pad.a, 0, 1))}


def outlet_speech(pad: PAD):
    """语音韵律读 A [DESIGN §4.3]（现有 TTS 已有情绪控制能力）。"""
    return {"rate": float(np.clip(1.0 + 0.35 * pad.a, 0.6, 1.8)),
            "pitch": float(np.clip(1.0 + 0.20 * pad.a + 0.08 * pad.p, 0.7, 1.5)),
            "energy": float(np.clip(1.0 + 0.30 * pad.a, 0.6, 1.6))}


def outlet_face(pad: PAD):
    """表情/姿态读 A 与 P。"""
    return {"au6": float(np.clip(0.5 + 0.5 * pad.p, 0, 1)),
            "au12": float(np.clip(0.5 + 0.4 * pad.p, 0, 1)),
            "au4": float(np.clip(0.5 - 0.5 * pad.p + 0.2 * pad.a, 0, 1)),
            "tension": float(np.clip(0.3 + 0.7 * pad.a, 0, 1))}


def outlet_memory_gate(pad: PAD, surprise=0.0, attention=0.0, cfg=None):
    """记忆写入门 in [0,1]：A × 惊讶 × 注意。

    ⚠ 只适用于情节/事件记忆（我经历过什么），不适用于事实记忆
    （世界的当前状态）——事实记忆按情绪挑着写在 E3/E5/E7 上 144/144 输给
    FIFO，连神谕都输 [DESIGN §4.3 补注]。writer 对 fact 一律全写。
    """
    cfg = cfg or {}
    wa, ws, wn = cfg.get("w_a", 0.4), cfg.get("w_s", 0.4), cfg.get("w_n", 0.2)
    gate = (wa * float(np.clip(pad.a, 0, 1))
            + ws * float(np.clip(surprise, 0, 1))
            + wn * float(np.clip(attention, 0, 1)))
    return float(np.clip(gate, 0.0, 1.0))


# ---------------------------------------------------------------- 估计量
def autocov(x):
    """序列的 C0/C1/C2（去均值）。x: (B,T) 或 (T,)。"""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    d = x - x.mean(axis=1, keepdims=True)
    c0 = (d * d).mean(axis=1)
    c1 = (d[:, 1:] * d[:, :-1]).mean(axis=1)
    c2 = (d[:, 2:] * d[:, :-2]).mean(axis=1)
    return c0, c1, c2


def estimate_theta_c2c1(series, dt):
    """a = C2/C1 -> theta = -ln(a)/dt；顺带 var_s = C1²/C2、观测噪声 r = C0 - var_s。

    返回 dict(a, theta, var_s, r, mu)。C1<=0 或 C2<=0 时返回 None（序列太短或无自相关）。
    """
    c0, c1, c2 = autocov(series)
    c0, c1, c2 = float(c0[0]), float(c1[0]), float(c2[0])
    if c1 <= 1e-12 or c2 <= 1e-12:
        return None
    a = float(np.clip(c2 / c1, 1e-4, 1 - 1e-6))
    var_s = max(c1 * c1 / c2, 1e-9)
    return dict(a=a, theta=-math.log(a) / dt, var_s=var_s, r=max(c0 - var_s, 1e-9))


def estimate_theta_raw(series, dt):
    """a = C1/C0（所有人默认写法，留作对照——它是有偏的）。"""
    c0, c1, _ = autocov(series)
    c0, c1 = float(c0[0]), float(c1[0])
    if c0 <= 1e-12:
        return None
    a = float(np.clip(c1 / c0, 1e-4, 1 - 1e-6))
    return dict(a=a, theta=-math.log(a) / dt, var_s=c0, r=1e-9)


# ---------------------------------------------------------------- 整体
@dataclass
class AffectSystem:
    dyn: OUDynamics = field(default_factory=OUDynamics)
    buf: ReactionBuffer = field(default_factory=ReactionBuffer)
    state: PAD = field(default_factory=PAD)
    t_ms: float = 0.0
    # 自己的生成器：调用方不传 rng 也不许碰 OS 熵（同种子同结果）
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(2026))
    log: List[dict] = field(default_factory=list)
    # 性格基线（成长层供数，mu_at = base_mu + 昼夜偏移）。None = 出厂 EMO_MU。
    base_mu: Optional[np.ndarray] = None
    # 时刻源：默认本地钟（昼夜节律从这里读「现在几点」）。测试注入固定
    # epoch 即得固定小时——确定性纪律在昼夜节律下的延伸：噪声进 seed，
    # 时刻进 clock，两者都可注入。
    clock: Callable[[], float] = time.time
    # 并发约定：state 只允许**整对象替换**（tick/decay/mean_revert 都是重新
    # 赋值 self.state），不许原地改字段——锁外的读者（web defer 闸门、前端
    # 轮询）靠这条拿到完整的新旧快照之一，不会撕裂读。要改字段先整体换。

    # ------------------------------------------------ 基线（性格 + 昼夜）
    def mu_at(self, hour=None):
        """此刻的完整基线 = 性格基线 + 昼夜偏移。

        mu 从此不再是一个常数：成长层定「平时什么底色」，昼夜节律定
        「这一天这个时刻偏向哪」，OU 负责让状态回到它。"""
        base = config.EMO_MU if self.base_mu is None else self.base_mu
        base = np.asarray(base, dtype=np.float64)
        if not rhythm.on():
            return base.copy()   # 拷贝防别名：谁原地改 dyn.mu 会污染模块级常量
        if hour is None:
            hour = rhythm.hour_of(self.clock())
        return base + rhythm.offset(hour)

    def refresh_mu(self, hour=None):
        """把 dyn.mu 刷成此刻的基线。每次动力学步进前调用——mu 是吸引子，
        吸引子本身随钟面缓慢移动，OU 会自然把状态拖向「这个时刻该有的底色」
        （凌晨唤醒被压下去、傍晚抬起来），不需要任何额外机制。"""
        self.dyn.mu = self.mu_at(hour)

    def set_base_mu(self, mu):
        """性格基线热更新（成长层塑形后调用）：特质动了 → 基线立刻生效。

        旧实现的断点：dyn.mu 只在开机时赋值一次，长会话里第 7 步塑形
        移动了特质后，行为偏置（face_bias）立刻变、情绪底色却要重启才变。
        现在基线统一从这里进，改完立即 refresh。"""
        self.base_mu = np.asarray(mu, dtype=np.float64)
        self.refresh_mu()

    def tick(self, dt_ms, apr: Optional[Appraisal] = None,
             rng: Optional[np.random.Generator] = None,
             drive: Optional[PAD] = None):
        self.t_ms += dt_ms
        self.refresh_mu()
        rng = rng or self.rng
        if drive is None:
            drive = PAD()
            if apr is not None:
                delta = appraise_to_pad(apr)
                if self.buf.offer(self.t_ms, delta):
                    drive = delta
        else:
            _ = apr
        drive = drive + self.buf.drain(self.t_ms)
        self.state = self.dyn.step(self.state, dt_ms / 60000.0, drive, rng)
        self._log(dt_ms)
        return dict(t_ms=self.t_ms, p=self.state.p, a=self.state.a,
                    d=self.state.d)

    def shock(self, delta: PAD):
        """情绪冲击：一次性评价事件（被骂/被夸/意外）全量落进状态，带饱和。

        为什么不走 drive 路径：drive 在 OU 步进里乘 (1-a)≈0.15 的阻尼——
        那是为 tick 内部的小 novelty 评价设计的（阻尼防抖）。但人的现象
        学是「被狠骂一记当场变脸，然后慢慢恢复」——冲击该一次到位，
        衰减交给之后的 decay。形态：shock（瞬时跳变）+ decay（慢恢复），
        正是人的情绪事件曲线。

        饱和（习惯化）：同方向已经拉满时，同向冲击效果递减——已经狂喜
        时再被夸不会更开心；反方向冲击不受影响（狂喜时被骂照样重）。
        逐维：effective = delta × (1 − sign(delta)·state)。
        """
        cur = self.state.as_array()
        dv = delta.as_array()
        sat = np.where(dv > 0, np.clip(cur, 0.0, 1.0), np.clip(-cur, 0.0, 1.0))
        eff = dv * (1.0 - sat)
        self.state = PAD.from_array(cur + eff).clipped()
        self.t_ms += 1000.0
        self._log(1000.0)
        return self.state

    def decay(self, dt_ms):
        """无外部冲击地流过 dt。**常驻时钟与墙钟同步都用它。**

        **不要切小步。** 上一版切成 60 秒一步，想「让衰减真的发生」，
        实测无效甚至更差（200 次重复，A 维初值 0.7）：

            间隔      精确OU应为   一步跨均值/sd     切步均值/sd
            10 分钟    0.1562     +0.1265 / 0.4728  +0.0789 / 0.4693
            20 分钟    0.0349     +0.0108 / 0.4838  +0.0089 / 0.5248

        原因是噪声按 sqrt(n) 累积，n 步之后总量和一步跨完全同阶，而均值
        还要被反复扰动。OU 无驱动时的条件期望本身就是确定性衰减（噪声的
        期望是 0），所以长时间无交互的正确做法是**一次均值回归**：
        不抽签、不攒噪声，A 维 0.7 过 10 分钟精确变成 0.156。

        代价：零。mean_revert 是一次 exp()。

        短间隔（≤ EMO_STEP_MS）仍走随机步——那时候人就在旁边，
        情绪该有点自然抖动。

        昼夜节律：两条路步进前都 refresh_mu——短间隔 mu 取此刻；长间隔
        mu 取**终点**时刻（理由见下方分支内注释）。
        """
        dt_ms = max(float(dt_ms), 0.0)
        if dt_ms <= 0.0:
            return self.state
        if dt_ms <= config.EMO_STEP_MS:
            self.refresh_mu()
            self.state = self.dyn.step(self.state, dt_ms / 60000.0,
                                       PAD(), self.rng)
        else:
            # 长间隔：一次均值回归到「此刻」的基线。mu 刷成终点时刻的——
            # 情绪记忆的半衰期只有分钟级（theta 0.10~0.15/min），昼夜偏移
            # 的时标是小时：任何跨峰谷的长间隔里 a=exp(-θ·dt)≪1，路径上
            # 的 mu 变化全被指数记忆抹掉，到达时状态只贴「这个时刻」的
            # 吸引子。分段积分能省下的误差实测 ≤0.01（试过按小时分段），
            # 远小于幅度 0.06~0.12 的设计容差——不值得添机器。
            # （对比上面切小步的教训：那是随机步的 sqrt(n) 噪声问题；这里
            # 是确定性映射，本就没有噪声可攒。）
            self.refresh_mu()
            self.state = self.dyn.mean_revert(self.state, dt_ms / 60000.0)
        self.t_ms += dt_ms
        self._log(dt_ms)
        return self.state

    def _log(self, dt_ms):
        """历史留 600 条（约 10 小时的一分钟粒度）。

        不设上界的话每次 tick 都 append 一条 dict，常年运行就是内存泄漏；
        但也不能一条不留——`estimate_theta_*` 与前端曲线都要读它。600 条
        足够拟合（c0/c1/c2 只需 3 个滞后），也够画出最近 10 小时的状态。
        """
        self.log.append(dict(t_ms=self.t_ms, dt=dt_ms, p=self.state.p,
                             a=self.state.a, d=self.state.d))
        if len(self.log) > 600:
            del self.log[:len(self.log) - 600]

    def outlets(self):
        return dict(language=outlet_language(self.state),
                    speech=outlet_speech(self.state),
                    face=outlet_face(self.state))

    def arousal(self):
        return float(np.clip(self.state.a, 0, 1))
