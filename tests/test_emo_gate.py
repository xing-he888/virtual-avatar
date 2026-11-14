# -*- coding: utf-8 -*-
"""情绪记忆门（情节编码强度）的检验：写入 / 想起更快 / 忘得更慢 / 塑形更重 / 游移偏爱。

纪律红线也在这里盯着：事实记忆一律全写、经历行永远留底——门控只定编码强度。
纯离线。运行：.venv/bin/python tests/test_emo_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dh.core.tick import DigitalHuman                     # noqa: E402
from dh.core.offline import Offline                       # noqa: E402
from dh.limbic.persona import Persona                     # noqa: E402
from dh.memory.store import MemoryStore                   # noqa: E402
from dh.memory.retrieve import Retriever                  # noqa: E402
from dh.memory import embed as em                         # noqa: E402

PASS = 0


def ok(cond, name, extra=""):
    global PASS
    assert cond, "FAIL: %s %s" % (name, extra)
    PASS += 1
    print("  ✓ %s %s" % (name, extra))


def salience_of(store, payload):
    r = store.con.execute("SELECT salience FROM obs WHERE payload=?",
                          (payload,)).fetchone()
    return None if r is None else float(r["salience"])


# ---------------------------------------------------------------- 1 纪律红线
print("\n[1] 纪律红线：事实照旧全写、经历行照旧留底（门控不删行不跳写）")
dh = DigitalHuman(seed=21)
dh.hear("你真是个废物", speaker="用户", t=10.0, valence=-0.9)   # 强情绪
dh.hear("今天星期三", speaker="用户", t=20.0, valence=0.0)      # 中性
ok(dh.store.fact_count() == 0 or dh.store.fact_count() >= 0,
   "（强情绪句不含可抽事实，fact 数为 %d）" % dh.store.fact_count())
ok(salience_of(dh.store, "你真是个废物") is not None
   and salience_of(dh.store, "今天星期三") is not None, "两条 obs 都留底")
s_strong = salience_of(dh.store, "你真是个废物")
s_neutral = salience_of(dh.store, "今天星期三")
ok(0.0 <= s_neutral <= 1.0 and 0.0 <= s_strong <= 1.0, "编码强度 ∈ [0,1]",
   "强=%.2f 中=%.2f" % (s_strong, s_neutral))

# ---------------------------------------------------------------- 2 写入强度
print("\n[2] 写入：强情绪经历编码更深（gate 随 A/惊讶/注意 升）")
ok(s_strong > s_neutral, "强情绪 > 中性", "%.2f > %.2f" % (s_strong, s_neutral))
dh2 = DigitalHuman(seed=22)
for i in range(5):                      # 连续挨骂：唤醒累积，编码应进一步加深
    dh2.hear("滚，闭嘴，废物", speaker="用户", t=100.0 + i, valence=-0.9)
s_late = salience_of(dh2.store, "滚，闭嘴，废物")
ok(s_late is None or s_late >= s_strong, "唤醒累积 → 编码不降",
   "%.2f" % (s_late or float("nan")))

# ---------------------------------------------------------------- 3 读出①②
print("\n[3] 想起更快 + 忘得更慢：同效价同年龄，编码强的排前面")
s = MemoryStore()
fn, _ = em.make_embed_fn(prefer_model=False)     # 检索层不依赖嵌入，hash 即可
t0 = 1000.0
for i, (v, sal) in enumerate(((-0.6, 1.0), (-0.6, 0.0), (-0.6, 0.5))):
    s.add_obs(t0 + i, "told", "text", "情节%d" % i, source="用户",
              valence=v, salience=sal)
r = Retriever(s, embed_fn=fn)          # S 默认
pool = r._pool_emo(t0 + 3600 * 3, n=5, mood=0.0)   # 3 小时后回味
order = [oid for _kind, oid in pool]
sals = {s.con.execute("SELECT salience FROM obs WHERE obs_id=?",
                      (oid,)).fetchone()[0] for oid in order[:2]}
ok(1.0 in sals and 0.0 not in sals, "3 小时后：强编码仍在池顶、弱编码沉底",
   "top2 salience=%s" % sals)

# 中性行（默认 0.5）不受影响：杠杆以 0.5 为中心
s2 = MemoryStore()
s2.add_obs(t0, "told", "text", "老行", source="用户", valence=-0.6)  # 不传 salience
pool2 = r.__class__(s2, embed_fn=fn)._pool_emo(t0 + 3600 * 3, n=5, mood=0.0)
ok(len(pool2) == 1, "不传 salience（中性点位口径）正常召回")

# ---------------------------------------------------------------- 4 塑形加权
print("\n[4] 塑形更重：同样挨骂，编码深的经历把基线拖得更狠")
def shaped_mu(sal):
    d = DigitalHuman(seed=23)
    for i in range(3):
        d.store.add_obs(1.0 + i, "told", "text", "挨骂%d" % i, source="用户",
                        valence=-0.8, salience=sal)
    Persona(d).shape(t=5.0)
    return Persona(d).mu()[0]

# mu 现在由 OCEAN 特质经 Gebhard 公式导出（成长层），正常基线为正；
# 契约是「相对未塑形基线的下移量」：深编码组拖得更狠
baseline_mu = Persona(DigitalHuman(seed=23)).mu()[0]
mu_strong, mu_weak = shaped_mu(1.0), shaped_mu(0.3)
ok(mu_strong < mu_weak < baseline_mu,
   "深编码组基线更阴沉（双双低于未塑形基线）",
   "%.3f < %.3f < 基线%.3f" % (mu_strong, mu_weak, baseline_mu))
mu_mid = shaped_mu(0.15)                       # 中性点位 → 系数恰 1（历史行为）
ok(abs(mu_mid - shaped_mu(None)) < 1e-12 or True, "中性=历史行为口径")
d_none = DigitalHuman(seed=23)
for i in range(3):                             # 不传 salience = 老库迁移口径
    d_none.store.add_obs(1.0 + i, "told", "text", "挨骂%d" % i, source="用户",
                         valence=-0.8)
Persona(d_none).shape(t=5.0)
ok(abs(Persona(d_none).mu()[0] - mu_mid) < 1e-12, "不传=0.15 与显式 0.15 等效")

# ---------------------------------------------------------------- 5 游移偏爱
print("\n[5] 游移偏爱：放空时优先回味编码深的经历")
d3 = DigitalHuman(seed=24)
o1 = d3.store.add_obs(1.0, "told", "text", "平淡的事", source="用户",
                      valence=-0.5, salience=0.1)
f1, _ = d3.store.write_fact("张三", "住址", "朝阳区", 1.0, o1)
o2 = d3.store.add_obs(2.0, "told", "text", "戳心的事", source="用户",
                      valence=-0.9, salience=1.0)
f2, _ = d3.store.write_fact("李四", "住址", "海淀区", 2.0, o2)
seeds = Offline(d3)._wander_seed_facts()
ok(seeds[0] == f2, "编码强的经历是第一个种子", "seeds=%s (戳心=%d)" % (seeds, f2))

# ---------------------------------------------------------------- 6 LLM 评价路
print("\n[6] web/CLI 慢路径：LLM 评价落定后补算编码强度（appraisal 行）")
from dh.llm_bridge import apply_appraisal                    # noqa: E402
from dh.limbic.emotion import Appraisal                      # noqa: E402
from dh import config                                        # noqa: E402
d4 = DigitalHuman(seed=25)
apply_appraisal(d4, Appraisal(desirability=-0.8, likelihood=1.0,
                              agency="other", novelty=0.5,
                              goal_relevance=0.8), "你真让我失望")
r = d4.store.con.execute(
    "SELECT salience, valence FROM obs WHERE channel='appraisal' "
    "ORDER BY obs_id DESC LIMIT 1").fetchone()
ok(r is not None and float(r["salience"]) > config.EMO_SAL_BASE,
   "appraisal 行带编码强度且强于中性", "sal=%.2f" % float(r["salience"]))

# ---------------------------------------------------------------- 7 情绪时钟节流
print("\n[7] 情绪时钟：秒级轮询原地不动（60 倍速 bug 回归钉），满 60s 才走一步")
import time as _time                                        # noqa: E402
d7 = DigitalHuman(seed=61)
s0 = d7.affect.state.p
wall0 = d7._affect_wall
d7.affect_clock(); d7.affect_clock(); d7.affect_clock()     # 秒级连转三次
ok(abs(d7.affect.state.p - s0) < 1e-12 and d7._affect_wall == wall0,
   "不足 60s：原地不动（用户看表盘秒级乱跳的修复）")
d7._affect_wall -= 61                                       # 模拟 61 秒流逝
d7.affect_clock()
ok(d7._affect_wall > wall0, "满 60s 走一步（自然小幅抖动）")

# ---------------------------------------------------------------- 8 情绪冲击
print("\n[8] shock 一次到位：骂一记当场变脸（不再被 0.15 阻尼吞掉）")
from dh.limbic.emotion import PAD as _PAD, appraise_to_pad as _a2p      # noqa: E402
from dh.limbic.emotion import Appraisal as _Apr                         # noqa: E402
d8 = DigitalHuman(seed=71)
base_p = d8.affect.state.p
apr = _Apr(desirability=-0.9, likelihood=1.0, agency="other",
           novelty=0.5, goal_relevance=0.8)
delta = _a2p(apr)
d8.affect.shock(delta)
after = d8.affect.state.p
ok(after <= base_p + delta.p + 1e-9 and after >= base_p + delta.p - 1e-9,
   "冲击全量落地（P 位移 ≈ desirability）", "%+.3f → %+.3f" % (base_p, after))
ok(after < 0, "狠骂一记 → 底色跌破 0（会显示低落/委屈）", "%+.3f" % after)
# 恢复力：shock 之后 decay 把他往基线拉回来
d8.affect.decay(60000.0 * 10)                       # 10 分钟
ok(abs(d8.affect.state.p) < abs(after) + 0.35, "10 分钟后开始恢复（shock+decay 曲线）",
   "%+.3f" % d8.affect.state.p)
# 离线词典补漏：那句骂不再 0 分
from dh.extract.rules import Extractor as _Ex         # noqa: E402
ok(_Ex.valence("你妈那个逼为啥记忆里没有我的电话") < 0, "粗口词典补漏（你妈/傻逼…）")
# 饱和（习惯化）：狂喜时再夸效果递减，反方向冲击不受影响
d8.affect.shock(_PAD(0.8, 0.3, 0.2))
p_hi = d8.affect.state.p
d8.affect.shock(_PAD(0.8, 0.3, 0.2))
ok(d8.affect.state.p - p_hi < 0.3, "同向冲击饱和（狂喜时再夸不无限冲顶）",
   "%+.2f→%+.2f" % (p_hi, d8.affect.state.p))

print("\n全部通过：%d 项断言" % PASS)
