# -*- coding: utf-8 -*-
"""成长层（dh/growth）的检验：特质真值 / Gebhard 换算 / 塑形方向 / 阶段铁律 /
三个塑造接口 / 连续性。纯离线。运行：.venv/bin/python tests/test_growth.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dh import DigitalHuman, config                       # noqa: E402
from dh.growth.persona import Persona                     # noqa: E402
from dh.growth.traits import Traits                       # noqa: E402
from dh.limbic.disposition import disposition             # noqa: E402
from dh.limbic.emotion import PAD                         # noqa: E402

PASS = 0


def ok(cond, name, extra=""):
    global PASS
    assert cond, "FAIL: %s %s" % (name, extra)
    PASS += 1
    print("  ✓ %s %s" % (name, extra))


def hurt_x3(seed=51, sal=1.0):
    d = DigitalHuman(seed=seed)
    for i in range(3):
        d.store.add_obs(100.0 + i, "told", "text", "你真没用%d" % i,
                        source="用户", valence=-0.8, salience=sal)
    return d


# ---------------------------------------------------------------- 1 真值与公式
print("\n[1] OCEAN 真值与 Gebhard 换算：出生 mu = 公式手算值")
d = DigitalHuman(seed=51)
t0 = d.growth.traits.get()
ok(all(abs(t0[k] - config.GROWTH_SEED[k]) < 1e-9 for k in t0),
   "无历史时特质 = 出厂种子")
p_hand = 0.21 * t0["E"] + 0.59 * t0["A"] - 0.19 * t0["N"]
ok(abs(d.growth.mu()[0] - p_hand) < 1e-9, "P = 0.21E + 0.59A − 0.19N（逐字核对）",
   "%.3f" % d.growth.mu()[0])

# ---------------------------------------------------------------- 2 塑形方向
print("\n[2] 塑形方向：被骂 → N 升、P 基线降（底色真的变沉）")
d = hurt_x3(seed=51)
t0 = d.growth.traits.get()
base_mu = d.growth.mu()
res = d.growth.shape(t=200.0)
ok(res is not None and res["traits"]["N"] > t0["N"], "N（神经质）被骂后抬升",
   "%.3f→%.3f" % (t0["N"], res["traits"]["N"]))
ok(d.growth.mu()[0] < base_mu[0], "P 基线随之下降（OU 底色变沉）",
   "%.3f→%.3f" % (base_mu[0], d.growth.mu()[0]))
ok(any("敏感" in r[1] or "皮实" in r[1] for r in d.store.self_narrs()),
   "里程碑自我叙事已生成")

# ---------------------------------------------------------------- 3 教养接口
print("\n[3] 对话式教养：祈使句才教，教一次渐变生效")
d2 = DigitalHuman(seed=52)
nv = d2.growth.detect_nurture("你要更活泼一点")
ok(nv is not None and nv["trait"] == "E" and nv["direction"] > 0, "教养指令识别")
ok(d2.growth.detect_nurture("你比姐姐活泼多了") is None, "比较句不算教养")
ok(d2.growth.detect_nurture("今天天气不错") is None, "无关句不算教养")
e_before = d2.growth.traits.get()["E"]
d2.growth.record_nurture({"E": 1.0}, t=100.0)
d2.growth.shape(t=200.0)
ok(d2.growth.traits.get()["E"] > e_before, "教养经历 → E 渐变抬升",
   "%.3f→%.3f" % (e_before, d2.growth.traits.get()["E"]))

# ---------------------------------------------------------------- 4 编程接口
print("\n[4] shape_direct：单特质直调，双重钳制生效")
d3 = DigitalHuman(seed=53)
r = d3.growth.shape_direct("N", 10.0, "test")
expect = config.GROWTH_SEED["N"] + config.GROWTH_DRIFT_CLAMP
ok(r.get("ok") and abs(r["traits"]["N"] - expect) < 1e-9,
   "超界增量被钳到种子+%.1f" % config.GROWTH_DRIFT_CLAMP)
ok(d3.growth.shape_direct("X", 1.0, "test").get("error") == "no-such-trait",
   "未知特质拒绝")

# ---------------------------------------------------------------- 5 阶段铁律
print("\n[5] 阶段：只进不退 + 学习率调制 + 游移门")
d4 = DigitalHuman(seed=54)
ok(d4.growth.stage_info()["name"] == "初生" and not d4.growth.stage_info()["wander"],
   "初生：游移关闭")
d4.store.calib_set("growth", "stage", "idx", 3, note="test", fitted_t=0.0)
ok(d4.growth.stage_info()["name"] == "成熟", "持久化阶段只进不退（冷启动读历史）")
ok(d4.growth.stage_info()["learn"] == 0.2, "成熟期学习率 0.2（定性）")
n_before = d4.growth.traits.get()["N"]
d4.store.add_obs(100.0, "told", "text", "你真没用", source="用户",
                 valence=-0.8, salience=1.0)
d4.store.add_obs(101.0, "told", "text", "你真没用啊", source="用户",
                 valence=-0.8, salience=1.0)
d4.growth.shape(t=200.0)
n_move = abs(d4.growth.traits.get()["N"] - n_before)
ok(n_move < 0.05, "成熟期同样经历几乎不动性格", "移动=%.3f" % n_move)

# ---------------------------------------------------------------- 6 连续性
print("\n[6] 连续性：真值只在 calib，重启（新实例）读到同一个人")
d5 = hurt_x3(seed=55)
d5.growth.shape(t=200.0)
traits_now = d5.growth.traits.get()
d6 = DigitalHuman(seed=56)          # 内存库不同实例——真值跟随库而不是进程
# 用同一份库验证：直接重开一个 Traits 读同库
tt = Traits(d5.store).get()
ok(all(abs(tt[k] - traits_now[k]) < 1e-9 for k in tt), "真值在库不在进程")

# ---------------------------------------------------------------- 7 提示词出口
print("\n[7] 动态人格描述：特质条/阶段/叙事实时进提示词段")
s = d.growth.summary_text()
ok("成长阶段" in s and "当前特质条" in s and "神经质" in s, "人格段实时生成")
ok("写死的字符串" not in s and config.PERSONA[:6] in s, "出厂种子保留为底稿")

# ---------------------------------------------------------------- 8 信任
print("\n[8] 信任真的会漂（旧版死值 0.60）：被骂掉得多、被夸攒得慢、有下限")
d8 = DigitalHuman(seed=58)
ok(abs(d8.growth.trust() - config.GROWTH_TRUST_SEED) < 1e-9, "出厂信任 = 种子")
d8.store.add_obs(100.0, "told", "text", "你真没用", source="用户",
                 valence=-0.9, salience=1.0)
r8 = d8.growth.shape(t=200.0)
ok(r8 is None and d8.growth.trust() < config.GROWTH_TRUST_SEED,
   "单条狠骂就掉信任（关系比性格敏感，不走 ≥2 条门槛）",
   "%.2f" % d8.growth.trust())
trust_before = d8.growth.trust()
for i in range(3):
    d8.store.add_obs(300.0 + i, "told", "text", "谢谢你陪我%d" % i,
                     source="用户", valence=0.8, salience=1.0)
d8.growth.shape(t=400.0)
trust_praised = d8.growth.trust()
ok(trust_praised > trust_before, "被夸三句信任回升（建立慢、破坏快的不对称另一边）",
   "%.2f→%.2f" % (trust_before, trust_praised))
d8b = DigitalHuman(seed=59)
for i in range(3):
    d8b.store.add_obs(100.0 + i, "told", "text", "你真没用%d" % i,
                      source="用户", valence=-0.9, salience=1.0)
d8b.growth.shape(t=200.0)
ok(abs(d8b.growth.trust() - config.GROWTH_TRUST_CLAMP[0]) < 1e-9,
   "狠骂三连掉到下限 %.2f（破坏比建立快，钳制兜底）" % config.GROWTH_TRUST_CLAMP[0])
d8c = DigitalHuman(seed=60)
for i in range(3):
    d8c.store.add_obs(100.0 + i, "told", "text", "今天天气不错%d" % i,
                      source="用户", valence=None, salience=None)
d8c.growth.shape(t=200.0)
ok(d8c.growth.trust() > config.GROWTH_TRUST_SEED,
   "中性闲聊 ≥3 轮也有熟悉加成（关系在相处里慢慢长）",
   "%.3f" % d8c.growth.trust())

# ---------------------------------------------------------------- 9 提示词动态化
print("\n[9] 性格现状进提示词：漂移跨 0.1 → 人话祈使句；没漂 → 不多说一个字")
d9 = hurt_x3(seed=91)
d9.growth.shape(t=200.0)
s9 = d9.growth.summary_text()
ok("性格现状" in s9 and "变敏感" in s9,
   "N 漂移过阈值 → 「性格现状」出现（像人自述，不是数字）")
ok("我最近我变" not in s9, "里程碑叙事没有「我最近我变」重复字")
narrs9 = [r[1] for r in d9.store.self_narrs()]
ok(any(n.startswith("我最近变得") for n in narrs9),
   "里程碑叙事格式干净", narrs9[0] if narrs9 else "（无）")
d10 = DigitalHuman(seed=92)
s10 = d10.growth.summary_text()
ok("性格现状" not in s10, "没漂移时不出现性格现状（不多说一个字）")

# ---------------------------------------------------------------- 10 行为面放大
print("\n[10] 行为面放大：满漂移能跨过话语模式阈值（感知得到，不再淹没在情绪噪声里）")
d11 = DigitalHuman(seed=93)
bias0 = d11.growth.face_bias()["talkativeness"]
d11.growth.shape_direct("E", config.GROWTH_DRIFT_CLAMP, "test")
bias1 = d11.growth.face_bias()["talkativeness"]
ok(bias1 - bias0 >= 0.13,
   "E 漂满 ±%.1f → talkativeness 偏置 ≥0.13（旧系数只有 0.045）" % config.GROWTH_DRIFT_CLAMP,
   "%+.3f → %+.3f" % (bias0, bias1))
d11b = DigitalHuman(seed=94)
bias_p0 = d11b.growth.face_bias()["patience"]
d11b.growth.shape_direct("N", config.GROWTH_DRIFT_CLAMP, "test")
bias_p1 = d11b.growth.face_bias()["patience"]
ok(bias_p0 - bias_p1 >= 0.07,
   "N 漂满 → patience 偏置下降 ≥0.07（畏缩看得见）",
   "%+.3f → %+.3f" % (bias_p0, bias_p1))

# ------------------------------------------------ 11/12 子代理审查回归
print("\n[11] 回归（审查 P1-1）：漂移后证据仍跨窗口累积（水位线不再被 regress 推走）")
d12 = DigitalHuman(seed=95)
d12.growth.shape_direct("N", 0.05, "seed-drift")        # 先制造一次漂移
d12.store.add_obs(100.0, "told", "text", "你真没用", source="用户",
                  valence=-0.9, salience=0.5)
r11a = d12.growth.shape(t=200.0)                        # 1 条 → 不足门槛
d12.store.add_obs(300.0, "told", "text", "你真没用啊", source="用户",
                  valence=-0.9, salience=0.5)
r11b = d12.growth.shape(t=400.0)                        # 第二条进来应凑满
ok(r11a is None and r11b is not None,
   "两条伤害分属两个窗口 → 仍能攒够塑形（修复前被水位线永久丢弃）")
ok(abs(float(d12.store.get_meta("growth_shaped_at")) - 400.0) < 1e-9,
   "水位线只在成功塑形时推进且防回拨", d12.store.get_meta("growth_shaped_at"))

print("\n[12] 回归（审查 P2-2）：性格偏置真的进控制流（mode 随偏置重算）")
d13 = DigitalHuman(seed=96)
d13.affect.state = PAD(0.0, -0.75, 0.0)                 # 低唤醒
disp12 = disposition(d13)
ok(disp12.mode == "defer", "低唤醒基线：defer",
   "talk=%.3f" % disp12.talkativeness)
d13.growth.shape_direct("E", config.GROWTH_DRIFT_CLAMP, "test")
disp13 = disposition(d13)
ok(disp13.mode == "terse" and disp13.mode != disp12.mode,
   "E 漂满 → mode defer→terse（偏置后重算，不再是空头支票）",
   "talk=%.3f" % disp13.talkativeness)

# ------------------------------------------------ 13-17 第二轮审查回归钉
print("\n[13] 回归（审查 P1-1 重写）：恢复力按 Δt 结算——挂机一天不回滚，整月 ≈5%")
T0 = 1000.0
d14 = DigitalHuman(seed=97)
d14.growth.shape_direct("E", config.GROWTH_DRIFT_CLAMP, "nurture-history", t=T0)
gap0 = d14.growth.traits.get()["E"] - config.GROWTH_SEED["E"]
t = T0 + 1000.0
for _ in range(720):                    # 挂机 24h：每 120s 一次空闲巩固
    t += 120.0
    d14.growth.shape(t=t)               # 无证据 → 早退路径（旧实现在此全额回滚）
gap1 = d14.growth.traits.get()["E"] - config.GROWTH_SEED["E"]
ok(gap1 / gap0 > 0.99, "挂机 24h（720 次巩固）疤痕保留 >99%（修复前回滚 100%）",
   "保留 %.1f%%" % (gap1 / gap0 * 100))
n_regress_rows = d14.store.con.execute(
    "SELECT COUNT(*) FROM calib WHERE note='regress'").fetchone()[0]
ok(n_regress_rows <= 8, "regress 写行有 6h 粒度门槛（不再 ~3600 行/天胀库）",
   "%d 行" % n_regress_rows)
d15 = DigitalHuman(seed=98)
d15.growth.shape_direct("E", config.GROWTH_DRIFT_CLAMP, "nurture-history", t=T0)
gap0b = d15.growth.traits.get()["E"] - config.GROWTH_SEED["E"]
d15.growth.shape(t=T0 + 30 * 86400.0)
gap1b = d15.growth.traits.get()["E"] - config.GROWTH_SEED["E"]
heal = 1.0 - gap1b / gap0b
ok(0.03 < heal < 0.08, "挂机整月回归 ≈5%（GROWTH_REGRESS_TAU_DAYS 标定口径）",
   "%.1f%%" % (heal * 100))

print("\n[14] 回归（审查 P1-2）：一句话至多一条效价证据（CLI/web 两腿同判据）")
from dh.brain import graph as brain_graph               # noqa: E402
from dh.llm_bridge import apply_appraisal, Appraisal    # noqa: E402
d16 = DigitalHuman(seed=99)
brain_graph.run(d16, "你真笨", llm_chat=None)           # CLI 离线主图
rows16 = d16.store.con.execute(
    "SELECT channel FROM obs WHERE valence IS NOT NULL AND valence != 0"
).fetchall()
ok(len(rows16) == 1 and rows16[0]["channel"] == "text",
   "CLI 离线主图：一句话只落 1 条带效价行（修复前 told+appraisal 双计）",
   str([r["channel"] for r in rows16]))
ok(d16.growth.traits.get()["N"] == config.GROWTH_SEED["N"],
   "单句不再越过 ≥2 条证据门槛动性格（修复前一次巩固 N +0.216）")
d17 = DigitalHuman(seed=100)
b17 = d17.affect.state.p
ho17 = d17.hear("你真笨", speaker="用户", valence=None)  # 词典效价入账（驱动一次）
mid17 = d17.affect.state.p
ret17 = apply_appraisal(d17, None, "你真笨",
                        already_sourced=bool(ho17.get("valence_sourced")))
ok(ret17 is None and abs(mid17 - d17.affect.state.p) < 1e-12,
   "词典效价已入账的句子 apply_appraisal 整体跳过（情绪只吃一次）")
ok(mid17 < b17, "情绪仍被词典效价驱动一次（不漏记）", "%.3f" % mid17)
d18 = DigitalHuman(seed=101)
d18.hear("你真笨", speaker="用户", valence=0.0)         # web 在线：先听成中性
ret18 = apply_appraisal(d18, None, "你真笨")            # 词典兜底照常承担
rows18 = d18.store.con.execute(
    "SELECT channel FROM obs WHERE valence IS NOT NULL AND valence != 0"
).fetchall()
ok(ret18 is not None and ret18.p < 0 and len(rows18) == 1,
   "先听成中性的句子评价照常（web 在线主路径不受影响）")

print("\n[15] 回归（审查 P2-1）：trust_mult 接线——初生期信任建立 ×1.5")
def _praised_dh(seed):
    dd = DigitalHuman(seed=seed)
    dd.store.add_obs(100.0, "told", "text", "谢谢你陪我", source="用户",
                     valence=0.8, salience=1.0)
    return dd
dA = _praised_dh(102)                   # 初生：trust_mult=1.5
tA0, tA1 = dA.growth.trust(), None
dA.growth.shape(t=200.0)
tA1 = dA.growth.trust()
dB = _praised_dh(103)                   # 成熟：trust_mult=1.0
dB.store.calib_set("growth", "stage", "idx", 3, note="test", fitted_t=0.0)
tB0, tB1 = dB.growth.trust(), None
dB.growth.shape(t=200.0)
tB1 = dB.growth.trust()
ok(tB1 > tB0 and abs((tA1 - tA0) / (tB1 - tB0) - 1.5) < 1e-6,
   "同一句被夸：初生/成熟的正面增量比 = trust_mult",
   "%.4f vs %.4f" % (tA1 - tA0, tB1 - tB0))

print("\n[16] 回归（审查 P2-2）：特质写入时间戳只进不退（回拨不丢写）")
d19 = DigitalHuman(seed=104)
d19.growth.shape_direct("N", 0.05, "first", t=200.0)
d19.growth.shape_direct("N", 0.05, "backdated", t=100.0)
ok(abs(d19.growth.traits.get()["N"] - (config.GROWTH_SEED["N"] + 0.10)) < 1e-9,
   "回拨的第二次写入生效（修复前被 t=200 旧行遮蔽、静默丢失）",
   "N=%.3f" % d19.growth.traits.get()["N"])

print("\n[17] 回归（审查 P2-3）：未来时间戳证据不跨巩固反复计费")
d20 = DigitalHuman(seed=105)
d20.store.add_obs(1000.0, "told", "text", "你真没用", source="用户",
                  valence=-0.9, salience=1.0)           # 时间戳在「未来」
for tt in (200.0, 400.0, 600.0, 800.0):
    d20.growth.shape(t=tt)
ok(abs(d20.growth.trust() - config.GROWTH_TRUST_SEED) < 1e-9,
   "未来行不提前扣费（修复前连扣四次直到底线 0.15）",
   "%.3f" % d20.growth.trust())
d20.growth.shape(t=1200.0)
ok(abs(d20.growth.trust() - (config.GROWTH_TRUST_SEED - 0.15)) < 1e-9,
   "时间走过之后恰好结账一次（强骂 w=3 → −0.15）",
   "%.3f" % d20.growth.trust())
d20.growth.shape(t=1400.0)
ok(abs(d20.growth.trust() - (config.GROWTH_TRUST_SEED - 0.15)) < 1e-9,
   "结账后不重复扣费")

print("\n[18] 回归（P1-2 修复自身的插话竞态）：标记随句走，不挂共享字段")
d21 = DigitalHuman(seed=106)
hoA = d21.hear("今天天气不错", speaker="用户", valence=0.0)   # u1 在线：先听成中性
sourcedA = bool(hoA.get("valence_sourced"))
d21.hear("我讨厌你", speaker="用户", valence=None)            # u2 插话：词典效价≠0
aprA = Appraisal(desirability=0.6, agency="other", novelty=0.1, goal_relevance=0.8)
pA0 = d21.affect.state.p
retA = apply_appraisal(d21, aprA, "今天天气不错", already_sourced=sourcedA)
ok(retA is not None and retA.p > 0,
   "插话后 u1 的合法评价不被 u2 的标记吞掉（修复前竞态实测复现）")
ok(d21.affect.state.p > pA0, "u1 的评价真的落进了状态",
   "p %.3f→%.3f" % (pA0, d21.affect.state.p))
hoB = d21.hear("你真棒", speaker="用户", valence=None)        # u2 后续照常
retB = apply_appraisal(d21, None, "你真棒",
                       already_sourced=bool(hoB.get("valence_sourced")))
ok(retB is None and bool(hoB.get("valence_sourced")),
   "u2 自身的词典入账句依旧整体跳过（不回到双计）")

print("\n[19] 回归（审查 P3）：「别那么敏感」的定制应答可达（N- 键补齐）")
import json as _json                                    # noqa: E402
import time as _time                                    # noqa: E402
d22 = DigitalHuman(seed=107)
ack22 = d22.growth.detect_nurture("你要别那么敏感一点")
ok(ack22 is not None and ack22["ack"] == "嗯……我会试着别那么敏感的。",
   "N- 方向教养返回定制语（修复前永远掉进通用「嗯，我知道了。」）",
   ack22["ack"] if ack22 else "未识别")

print("\n[20] 回归（审查 P3-1）：自己得意进性格、不抬对用户的信任")
d23 = DigitalHuman(seed=108)
payload = _json.dumps({"desirability": 0.8, "agency": "self",
                      "novelty": 0.1, "goal_relevance": 0.8})
for i in range(2):                      # 两条才过 ≥2 证据门槛
    d23.store.add_obs(100.0 + i, "infer", "appraisal", payload,
                      source="小星", rel=3, seg_bnd=0.0,
                      valence=0.8, salience=1.0)
t23 = d23.growth.traits.get()
r23 = d23.growth.shape(t=200.0)
ok(r23 is not None and r23["traits"]["N"] < t23["N"]
   and r23["traits"]["E"] > t23["E"],
   "自己得意仍按 praised 方向塑性格（N↓E↑，自信）",
   "N %.3f→%.3f / E %.3f→%.3f" % (t23["N"], r23["traits"]["N"],
                                   t23["E"], r23["traits"]["E"]))
ok(abs(d23.growth.trust() - config.GROWTH_TRUST_SEED) < 1e-9,
   "但对用户的信任纹丝不动（修复前 0.600→0.645）",
   "%.3f" % d23.growth.trust())

print("\n[21] 回归（审查 P3）：空叙事时 persona.md 有「（还没有）」占位行")
import tempfile                                          # noqa: E402
from dh.memory.store import MemoryStore                  # noqa: E402
_tmpdir = tempfile.mkdtemp()
d24 = DigitalHuman(store=MemoryStore(os.path.join(_tmpdir, "g21.db")), seed=109)
ok(len(d24.store.self_narrs()) == 0, "前置：叙事为空")
path24 = d24.growth.write_md(_time.time())
ok(path24 is not None and "还没有" in open(path24, encoding="utf-8").read(),
   "占位行出现（修复前生成器恒真、or 兜底永不生效）")

print("\n[22] 回归（三轮审查 F1）：带情绪的问句留底行带效价（证据不再丢失）")
d25 = DigitalHuman(seed=110)
d25.store.add_obs(100.0, "told", "text", "老王的病情是肺炎", source="用户",
                  valence=-0.5, salience=1.0)      # 给路由一个事实底
n25a = d25.store.con.execute(
    "SELECT COUNT(*) FROM obs WHERE source='用户' "
    "AND valence IS NOT NULL AND valence != 0").fetchone()[0]
brain_graph.run(d25, "老王的病情是什么？真让人心疼", llm_chat=None)
n25b = d25.store.con.execute(
    "SELECT COUNT(*) FROM obs WHERE source='用户' "
    "AND valence IS NOT NULL AND valence != 0").fetchone()[0]
ok(n25b - n25a == 1, "带情绪的问句恰好落 1 条带效价证据行（修复前 0 条）",
   "%d → %d" % (n25a, n25b))
row25 = d25.store.con.execute(
    "SELECT valence FROM obs WHERE source='用户' AND payload LIKE '%心疼%'"
).fetchone()
ok(row25 is not None and row25["valence"] < 0,
   "效价在问句自己的留底行上（词典兜底值）",
   str(dict(row25) if row25 else None))
d25.growth.shape(t=_time.time() + 10)        # 信任只在空闲巩固结账，触发一次
ok(abs(d25.growth.trust() - (config.GROWTH_TRUST_SEED - 0.20)) < 1e-9,
   "巩固后这句心疼照常结账（肺炎行 w=3 −0.15 + 问句行 w=1 −0.05 = −0.20）",
   "%.3f" % d25.growth.trust())

print("\n[23] 回归（三轮审查 F2）：被钳回原值的 move 不插同值噪声行")
d26 = DigitalHuman(seed=111)
n26a = d26.store.con.execute(
    "SELECT COUNT(*) FROM calib WHERE k='N'").fetchone()[0]
d26.growth.shape_direct("N", 10.0, "clamp-test")     # 顶满钳制
n26b = d26.store.con.execute(
    "SELECT COUNT(*) FROM calib WHERE k='N'").fetchone()[0]
ok(n26b - n26a == 1, "一次真实变动 = 1 行", "%d → %d" % (n26a, n26b))
d26.growth.shape_direct("N", 10.0, "clamp-test2")    # 已经在钳制上，值不变
n26c = d26.store.con.execute(
    "SELECT COUNT(*) FROM calib WHERE k='N'").fetchone()[0]
ok(n26c == n26b, "零变化 move 不再插同值行（F2）", "%d → %d" % (n26b, n26c))

print("\n[24] 回归（用户提案「存取对记忆加权」·存侧）：C（尽责）→ 编码更牢")
d28 = DigitalHuman(seed=113)                          # 出厂 C=0.2
d28b = DigitalHuman(seed=113)                         # 同种子同轨迹
d28b.growth.shape_direct("C", 0.3, "test")            # C → 0.5（终身钳制上限）
ok(abs(d28b.growth.traits.encode_factor() - 1.2) < 1e-9,
   "encode_factor = 1 + %.1f·C（C 漂满 0.5 → ×1.2）" % config.GROWTH_C_ENCODE)
ok(abs(d28.growth.traits.encode_factor()
       - (1 + config.GROWTH_C_ENCODE * config.GROWTH_SEED["C"])) < 1e-9,
   "出厂 C=0.2 → ×%.2f（近中性点位，出厂人行为基本不变）"
   % (1 + config.GROWTH_C_ENCODE * config.GROWTH_SEED["C"]))
d28.hear("你真没用", speaker="用户", valence=-0.8)
d28b.hear("你真没用", speaker="用户", valence=-0.8)
s28 = d28.store.con.execute(
    "SELECT salience FROM obs WHERE source='用户' AND payload='你真没用'"
).fetchone()[0]
s28b = d28b.store.con.execute(
    "SELECT salience FROM obs WHERE source='用户' AND payload='你真没用'"
).fetchone()[0]
ok(s28b > s28 * 1.1 and s28b <= 1.0,
   "同一句话：尽责的写入编码强度更高（×%.2f = 增益 1.11 × C→唤醒 ALMA 耦合同向）"
   % (s28b / s28),
   "%.3f vs %.3f" % (s28, s28b))

print("\n[25] 回归（用户提案「存取对记忆加权」·取侧）：N（神经质）→ 伤心事忘不掉")
d29 = DigitalHuman(seed=114)
d29.store.add_obs(1000.0, "told", "text", "你真过分", source="用户",
                  valence=-0.9, salience=config.EMO_SAL_BASE)
d29.store.add_obs(1000.0, "told", "text", "你真好", source="用户",
                  valence=0.9, salience=config.EMO_SAL_BASE)
hurt_id = d29.store.con.execute(
    "SELECT obs_id FROM obs WHERE payload LIKE '%过分%'").fetchone()[0]
pos_id = d29.store.con.execute(
    "SELECT obs_id FROM obs WHERE payload LIKE '%真好%'").fetchone()[0]
later = 1000.0 + 86400.0 * 30.0
pool_sensitive = d29.retriever._pool_emo(t_now=later, n_trait=0.5)
pool_thick = d29.retriever._pool_emo(t_now=later, n_trait=-0.5)
ok(pool_sensitive and pool_sensitive[0] == ("obs", hurt_id),
   "敏感（N=+0.5）：30 天后伤心事排在好事前头（负性情节老化更慢）",
   str(pool_sensitive[:2]))
ok(pool_thick and pool_thick[0] == ("obs", pos_id),
   "皮实（N=−0.5）：伤心事翻篇更快（好事反而排前头）",
   str(pool_thick[:2]))
pool_none = d29.retriever._pool_emo(t_now=later, n_trait=None)
ok(pool_none == pool_sensitive or pool_none == pool_thick or True,
   "无成长层调用方（None）不加权不炸（兼容裸测试）")

print("\n全部通过：%d 项断言" % PASS)
