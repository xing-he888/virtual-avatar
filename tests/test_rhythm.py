# -*- coding: utf-8 -*-
"""昼夜节律 + 基线热更新的检验：极值查表 / 有界 / 方向 / 隔夜分段回归 /
确定性 / DigitalHuman 集成（热更新不再等重启）。

纯离线。运行：.venv/bin/python tests/test_rhythm.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np                                              # noqa: E402

from dh import DigitalHuman, config                             # noqa: E402
from dh.limbic import rhythm                                    # noqa: E402
from dh.limbic.emotion import AffectSystem, PAD                 # noqa: E402

PASS = 0


def ok(cond, name, extra=""):
    global PASS
    assert cond, "FAIL: %s %s" % (name, extra)
    PASS += 1
    print("  ✓ %s %s" % (name, extra))


# ---------------------------------------------------------------- 1 纯函数极值
print("\n[1] 昼夜偏移：极值可查表（h=peak→+amp，h=peak+12→−amp）、有界、周期")
amp, peak = config.EMO_DIURNAL_AMP, config.EMO_DIURNAL_PEAK
for i in range(3):
    ok(abs(rhythm.offset(peak[i])[i] - amp[i]) < 1e-12,
       "P/A/D[%d] 峰时刻取 +amp" % i, "%.3f" % rhythm.offset(peak[i])[i])
    ok(abs(rhythm.offset(peak[i] + 12.0)[i] + amp[i]) < 1e-12,
       "P/A/D[%d] 谷时刻取 −amp" % i)
mx = max(abs(rhythm.offset(h / 4.0)[j]) for h in range(96) for j in range(3))
ok(mx <= amp.max() + 1e-12, "24h 逐刻采样 |offset| ≤ amp", "max=%.4f" % mx)
ok(all(abs(rhythm.offset(h)[j] - rhythm.offset(h + 24)[j]) < 1e-12
       for h in (0.0, 7.5, 13.25) for j in range(3)), "offset(h+24)=offset(h)")

# ---------------------------------------------------------------- 2 基线合成
print("\n[2] mu_at = 性格基线 + 昼夜偏移；开关关掉退回纯性格")
fixed = 1700000000.0
a = AffectSystem(clock=lambda: fixed)               # 固定钟：同种子同结果
a.base_mu = np.array([0.4, 0.1, 0.2])
h = rhythm.hour_of(fixed)
ok(np.allclose(a.mu_at(), a.base_mu + rhythm.offset(h), atol=1e-12),
   "mu_at(钟面) = base + offset(hour_of(钟))")
a.refresh_mu()
ok(np.allclose(a.dyn.mu, a.mu_at(), atol=1e-12), "refresh_mu 写进 dyn.mu")
on = config.EMO_DIURNAL_ON
try:
    config.EMO_DIURNAL_ON = False
    ok(np.allclose(a.mu_at(), a.base_mu, atol=1e-12), "开关关 → 纯性格基线")
finally:
    config.EMO_DIURNAL_ON = on

# ---------------------------------------------------------------- 3 方向
print("\n[3] 方向：凌晨谷、傍晚峰（与 config 出处定性一致）")
a2 = AffectSystem(clock=lambda: fixed)
a2.base_mu = np.zeros(3)
ok(a2.mu_at(hour=4.0)[1] < a2.mu_at(hour=17.0)[1],
   "唤醒基线：凌晨4点 < 傍晚17点",
   "%.3f < %.3f" % (a2.mu_at(hour=4.0)[1], a2.mu_at(hour=17.0)[1]))
ok(a2.mu_at(hour=4.0)[0] < a2.mu_at(hour=15.0)[0],
   "效价基线：凌晨4点 < 午后15点")

# ---------------------------------------------------------------- 4 隔夜同步
print("\n[4] 隔夜/中段同步：长间隔回归用「终点时刻」的基线（记忆只贴此刻吸引子）")
ts_end = time.mktime((2026, 3, 15, 8, 0, 0, 0, 0, -1))
dt_ms = 10.0 * 3600.0 * 1000.0                      # 10 小时（22:00→08:00 那类）
a3 = AffectSystem(clock=lambda: ts_end)
a3.base_mu = np.array([0.4, 0.1, 0.2])
a3.state = PAD(0.5, 0.5, 0.5)
a3.decay(dt_ms)
h_end = rhythm.hour_of(ts_end)
h_start = rhythm.hour_of(ts_end - dt_ms / 1000.0)
mu_end, mu_start = a3.mu_at(hour=h_end), a3.mu_at(hour=h_start)
d_end = float(np.abs(a3.state.as_array() - mu_end).max())
d_start = float(np.abs(a3.state.as_array() - mu_start).max())
ok(d_end < 1e-6, "10h 后逐维贴「今早」基线（a=exp(-θ·600)≈0）", "距=%.2e" % d_end)
ok(d_end < d_start, "贴今早而不是昨晚", "今早%.4f vs 昨晚%.4f" % (d_end, d_start))

# 中段（30 分钟）：终点 mu 语义可精确手算——确定性映射逐位验证
ts30 = time.mktime((2026, 3, 15, 14, 0, 0, 0, 0, -1))
a4 = AffectSystem(clock=lambda: ts30)
a4.base_mu = np.array([0.4, 0.1, 0.2])
a4.state = PAD(0.5, 0.5, 0.5)
a4.decay(30.0 * 60.0 * 1000.0)
mu30 = a4.mu_at(hour=rhythm.hour_of(ts30))
th = a4.dyn.theta
expect = mu30 + (np.array([0.5, 0.5, 0.5]) - mu30) * np.exp(-th * 30.0)
ok(float(np.abs(a4.state.as_array() - expect).max()) < 1e-9,
   "30 分钟回归 = mu_end + (s−mu_end)·exp(-θ·30)，逐位手算吻合",
   "差=%.2e" % float(np.abs(a4.state.as_array() - expect).max()))

# ---------------------------------------------------------------- 5 确定性
print("\n[5] 确定性：同钟同种子，两次演化逐位一致")


def run_sequence():
    s = AffectSystem(clock=lambda: fixed)
    s.base_mu = np.array([0.3, 0.0, -0.1])
    s.refresh_mu()
    s.decay(7.5 * 3600.0 * 1000.0)
    s.decay(30000.0)
    s.tick(1000.0, drive=PAD(0.2, 0.1, 0.0))
    return s.state


st1, st2 = run_sequence(), run_sequence()
ok((st1.p, st1.a, st1.d) == (st2.p, st2.a, st2.d),
   "逐位一致（噪声进 seed，时刻进 clock，两者皆可注入）")

# ---------------------------------------------------------------- 6 集成
print("\n[6] DigitalHuman 集成：基线来自成长层；塑形后热更新（不等重启）")
d = DigitalHuman(seed=61)
ok(np.allclose(d.affect.base_mu, d.growth.mu(), atol=1e-12),
   "开机 base_mu = 成长层 mu")
ok(np.allclose(d.affect.dyn.mu, d.affect.mu_at(), atol=1e-12),
   "开机 dyn.mu = base + offset(现在)")
p_before = float(d.affect.base_mu[0])
d.growth.shape_direct("N", 0.2, "rhythm-test")
ok(float(d.affect.base_mu[0]) < p_before,
   "shape_direct 后基线立刻下降（N↑ → P↓，热更新）",
   "%.3f→%.3f" % (p_before, d.affect.base_mu[0]))
ok(math.isclose(float(d.affect.dyn.mu[0]),
                float(d.affect.base_mu[0]) + float(rhythm.offset(
                    rhythm.hour_of(time.time()))[0]), abs_tol=1e-9),
   "dyn.mu 跟着新基线 + 当前时刻偏移")

print("\n全部通过：%d 项" % PASS)
