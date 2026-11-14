# -*- coding: utf-8 -*-
"""写入侧值等价的检验：词典归一 / 余弦合并 / 变更否决 / 豁免 / 传旧话 / 标定流。

纯离线（bge 本地模型，不调任何 API）。运行：
    .venv/bin/python tests/test_equiv_write.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dh.memory.store import MemoryStore                    # noqa: E402
from dh.memory.writer import Writer                        # noqa: E402
from dh.memory import embed as em                          # noqa: E402
from dh.memory.equiv import ValueEquivalence               # noqa: E402
from dh.extract.rules import normalize_value               # noqa: E402
from dh.extract.llm import Candidate                       # noqa: E402

PASS = 0


def ok(cond, name, extra=""):
    global PASS
    assert cond, "FAIL: %s %s" % (name, extra)
    PASS += 1
    print("  ✓ %s %s" % (name, extra))


def make(bge=True):
    s = MemoryStore(":memory:")
    for p, m in (("住址", 0), ("病情", 0), ("电话", 0), ("备用电话", 1)):
        s.declare_pred(p, m)
    if bge:
        fn, _ = em.make_embed_fn()
    else:
        fn, _ = em.make_embed_fn(prefer_model=False)
    return s, Writer(s, embed_fn=fn)


def cand(sub, pred, val, conf=0.9):
    return Candidate(subject=sub, predicate=pred, value=val, conf=conf,
                     tense="current")


def llm_write(s, w, text, t, cands, speaker="用户"):
    oid = s.add_obs(t, "told", "text", text, source=speaker)
    return w.hear_llm_candidates(text, t, speaker, oid, cands)


# ---------------------------------------------------------------- 1 词典归一
print("\n[1] normalize_value：Tier 0 与 Tier 1 同一张表")
ok(normalize_value("病情", "进医院") == "住院", "进医院→住院")
ok(normalize_value("病情", "出院了") == "出院", "出院了→出院")
ok(normalize_value("住址", "海淀") == "海淀", "非词典谓词原样")

# ---------------------------------------------------------------- 2 词典等价（零余弦）
print("\n[2] 词典等价：住院/进医院 余弦只有 0.719，靠词典接住（双向）")
s, w = make()
r1 = llm_write(s, w, "老王老婆住院了", 1.0, [cand("老王老婆", "病情", "住院")])
r2 = llm_write(s, w, "老王老婆进医院了", 2.0, [cand("老王老婆", "病情", "进医院")])
ok(r1["actions"] == ["new"] and r2["actions"] == ["confirm_equiv"],
   "第二次写进医院 → confirm_equiv", str(r2["actions"]))
ok(s.fact_count() == 1, "fact 只有一行", "n=%d" % s.fact_count())
ok(s.value_alias_lookup("老王老婆", "病情", "进医院") is not None, "别名已登记")
row = s.con.execute("SELECT valid_from,confirmed_t FROM fact").fetchone()
ok(row["valid_from"] == 1.0 and row["confirmed_t"] == 2.0,
   "valid_from 保留、confirmed_t 前移", str(dict(row)))

# 反方向：存侧先到的是未归一 raw（进医院），来侧是 Tier 0 规范值（住院）
s, w = make()
r1 = llm_write(s, w, "老王老婆进医院了", 1.0, [cand("老王老婆", "病情", "进医院")])
r2 = llm_write(s, w, "听说老王老婆住院了", 2.0, [cand("老王老婆", "病情", "住院")])
ok(r1["actions"] == ["new"], "raw 值先到 → new")
ok(r2["actions"] == ["confirm_equiv"], "住院 ↔ 已存的进医院（归一同串）→ 合并",
   str(r2["actions"]))
ok(s.fact_count() == 1, "不产生第二行", "n=%d" % s.fact_count())
ok(w.streams["老王老婆|病情"].phi == 0.0, "标定流吃规范值，phi=0",
   "phi=%.2f" % w.streams["老王老婆|病情"].phi)

# 出院仍是真实变更（单值互斥语义不受等价合并影响）
r3 = llm_write(s, w, "老王老婆出院了", 3.0, [cand("老王老婆", "病情", "出院")])
ok(r3["actions"] == ["new"], "出院 → new（真实变更照常 supersede）",
   str(r3["actions"]))
ok([x["value"] for x in s.active_values("老王老婆", "病情")] == ["出院"],
   "active 只有出院")

# ---------------------------------------------------------------- 3 余弦合并
print("\n[3] 余弦合并：海淀 是 海淀区 的另一种说法（value-only cos=0.897）")
s, w = make()
llm_write(s, w, "老王搬到海淀区了", 1.0, [cand("老王", "住址", "海淀区")])
r = llm_write(s, w, "老王搬海淀去了", 2.0, [cand("老王", "住址", "海淀")])
ok(r["actions"] == ["confirm_equiv"], "海淀 → confirm_equiv", str(r["actions"]))
ok(s.fact_count() == 1, "不产生第二行", "n=%d" % s.fact_count())
ok([x["value"] for x in s.active_values("老王", "住址")] == ["海淀区"],
   "fact 层存规范值")
ok(s.value_alias_lookup("老王", "住址", "海淀")["canonical"] == "海淀区",
   "表面形式进别名表")

# ---------------------------------------------------------------- 4 变更否决
print("\n[4] 变更信号词一票否决：搬到海淀区北苑 ≠ 海淀区（cos 0.837 但语义是搬家）")
s, w = make()
llm_write(s, w, "老王住在海淀区", 1.0, [cand("老王", "住址", "海淀区")])
eq = w.equiv.find("老王", "住址", "海淀区北苑")
ok(eq is None, "裸余弦接不住北苑（0.837<0.85）", str(eq))
r = llm_write(s, w, "老王搬到海淀区北苑了", 2.0,
              [cand("老王", "住址", "海淀区北苑")])
ok(r["actions"] == ["new"], "带「搬到」→ 跳过裁决直接 new", str(r["actions"]))
ok([x["value"] for x in s.active_values("老王", "住址")] == ["海淀区北苑"],
   "真实变更正常落位")

# ---------------------------------------------------------------- 5 豁免
print("\n[5] 豁免：数字值与电话谓词永不语义合并")
s, w = make()
llm_write(s, w, "老王的电话是13800001234", 1.0, [cand("老王", "电话", "13800001234")])
ok(w.equiv.find("老王", "电话", "13800001235") is None, "电话谓词豁免")
r = llm_write(s, w, "老王换号了，13800005678", 2.0, [cand("老王", "电话", "13800005678")])
ok(r["actions"] == ["new"], "换号是真实变更")
r2 = llm_write(s, w, "老王备用号 13900005678", 3.0,
               [cand("老王", "备用电话", "13900005678")])
r3 = llm_write(s, w, "老王备用号 139 0000 5678", 4.0,
               [cand("老王", "备用电话", "139 0000 5678")])
ok(r2["actions"] == ["new"] and r3["actions"] == ["new"],
   "数字串值豁免（多值并存维持字面判定）", str(r2["actions"] + r3["actions"]))

# ---------------------------------------------------------------- 6 传旧话不破
print("\n[6] 字面四分支原样保留：confirm / confirm_old / 复活")
s, w = make()
llm_write(s, w, "老王住在朝阳区", 1.0, [cand("老王", "住址", "朝阳区")])
r = llm_write(s, w, "老王住在朝阳区", 2.0, [cand("老王", "住址", "朝阳区")])
ok(r["actions"] == ["confirm"], "字面相同 → confirm")
llm_write(s, w, "老王搬海淀去了", 3.0, [cand("老王", "住址", "海淀")])   # new，supersede 朝阳
r = llm_write(s, w, "老王住在海淀区", 4.0, [cand("老王", "住址", "海淀区")])
ok(r["actions"] == ["confirm_equiv"], "复述现值（余弦路：海淀→海淀区 0.897）",
   str(r["actions"]))
r = llm_write(s, w, "听说老王还住朝阳区", 5.0, [cand("老王", "住址", "朝阳区")])
ok(r["actions"] == ["confirm_old"], "传旧话（失效值字面命中）→ 不复活",
   str(r["actions"]))
r = llm_write(s, w, "老王从海淀搬到朝阳区了", 6.0, [cand("老王", "住址", "朝阳区")],
              speaker="老王")
ok(r["actions"] == ["revive"], "变更信号+本人口径 → 复活（revive_ok）",
   str(r["actions"]))

# ---------------------------------------------------------------- 7 标定流
print("\n[7] 标定流吃规范值：表面形式不再虚增翻转率")
s, w = make()
for i, v in enumerate(("海淀区", "海淀", "海淀区", "海淀")):
    llm_write(s, w, "老王住在%s" % v, float(i) + 1, [cand("老王", "住址", v)])
st = w.streams["老王|住址"]
ok(st.phi == 0.0, "phi=0（旧实现这里是 1.0）", "phi=%.2f" % st.phi)

# ---------------------------------------------------------------- 8 hash 降级
print("\n[8] 嵌入模型不是 bge 时余弦层停用（无标定阈值不开合）")
s, w = make(bge=False)
ok(w.equiv.cos_enabled is False, "hash 口径 cos_enabled=False")
r = llm_write(s, w, "老王搬到海淀区了", 1.0, [cand("老王", "住址", "海淀区")])
r = llm_write(s, w, "老王搬海淀去了", 2.0, [cand("老王", "住址", "海淀")])
ok(r["actions"] == ["new"], "回到字面判定（旧行为）", str(r["actions"]))
ok(normalize_value("病情", "进医院") == "住院", "词典层仍然生效")

# ---------------------------------------------------------------- 9 别名短路
print("\n[9] 别名表命中后不再算余弦（第二次同表面形式微秒确认）")
s, w = make()
llm_write(s, w, "老王搬到海淀区了", 1.0, [cand("老王", "住址", "海淀区")])
llm_write(s, w, "老王搬海淀去了", 2.0, [cand("老王", "住址", "海淀")])
calls = {"n": 0}
orig = w.embed_fn


def counting(t):
    calls["n"] += 1
    return orig(t)


w.embed_fn = counting
w.equiv.embed_fn = counting
r = llm_write(s, w, "老王搬海淀去了", 3.0, [cand("老王", "住址", "海淀")])
ok(r["actions"] == ["confirm_equiv"], "别名直接命中")
calls["n"] = 0
eq = w.equiv.find("老王", "住址", "海淀")
ok(eq is not None and eq[1] == "海淀区", "find 走别名表")
ok(calls["n"] == 0, "别名路零次嵌入调用", "n=%d" % calls["n"])

# ---------------------------------------------------------------- 10 否定与前沿否决
print("\n[10] 否定语境不归一；前缀包含且差 >=2 字确定性否决")
ok(normalize_value("病情", "根本没进医院") == "根本没进医院",
   "「没进医院」不归一成住院")
s, w = make()
r = llm_write(s, w, "老王根本没进医院", 1.0, [cand("老王", "病情", "根本没进医院")])
ok(r["actions"] == ["new"], "否定句的值不合并进住院", str(r["actions"]))
eq = w.equiv.find("老王", "工作单位", "协和医院西院")
_ = eq  # 需要先有 协和医院 行才可比；这里直接走完整链
llm_write(s, w, "老王在协和医院上班", 2.0, [cand("老王", "工作单位", "协和医院")])
eq = w.equiv.find("老王", "工作单位", "协和医院西院")
ok(eq is None, "前缀否决：协和医院/协和医院西院（cos 0.862）不合", str(eq))
eq = w.equiv.find("老王", "住址", "海淀区北苑")
_ = eq
llm_write(s, w, "老王住在海淀区", 3.0, [cand("老王", "住址", "海淀区")])
ok(w.equiv.find("老王", "住址", "海淀区北苑") is None, "北苑同样被前缀否决")

print("\n全部通过：%d 项断言" % PASS)
