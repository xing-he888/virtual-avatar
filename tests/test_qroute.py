# -*- coding: utf-8 -*-
"""问句路由器的检验：L0 属主消解 / L1 语义路由 / L2 兜底与自学习 / 门槛修复。

纯离线（LLM 用 Mock，只验「老师教一次、路由器学会」的回路）。
运行：.venv/bin/python tests/test_qroute.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dh import DigitalHuman                               # noqa: E402
from dh import config                                     # noqa: E402
from dh.context import collect_context                    # noqa: E402
from dh.llm_bridge import llm_route_question              # noqa: E402

PASS = 0


def ok(cond, name, extra=""):
    global PASS
    assert cond, "FAIL: %s %s" % (name, extra)
    PASS += 1
    print("  ✓ %s %s" % (name, extra))


# ---------------------------------------------------------------- 1 L0
print("\n[1] L0：种子线索直路由 + 「我」→属主消解")
dh = DigitalHuman(seed=31)
r = dh.qrouter.route("老王现在电话是多少")
ok(r is not None and r["subject"] == "老王" and r["predicate"] == "电话"
   and r["via"] == "t0", "原小抄路径保留（实体+种子谓词）")
dh.store.set_owner("小明")
r = dh.qrouter.route("我电话多少")
ok(r is not None and r["subject"] == "小明" and r["predicate"] == "电话",
   "「我」消解到属主，谓词名命中", str((r["subject"], r["predicate"])))
ok(dh.qrouter.route("我今年100岁") is None, "陈述句不路由（交给写入侧）")
ok(dh.qrouter.route("今天天气如何") is None, "无关问句不硬路由（阈值拒绝）")

# 无属主时自指问句拒路由（宁可漏）
dh2 = DigitalHuman(seed=32)
ok(dh2.qrouter.route("我电话多少") is None, "无属主 → 自指问句拒路由（宁可漏记）")

# ---------------------------------------------------------------- 2 L1 自学习
print("\n[2] L1 自学习：学一条「年龄」线索，整族问法全部命中")
ok(dh.qrouter.route("我多少岁") is None, "学习前：年龄无线索 → 路由不出")
ok(dh.qrouter.learn("年龄", "多大年纪", source="llm") is not None, "L2 回填一条线索")
r = dh.qrouter.route("我多少岁")
ok(r is not None and r["subject"] == "小明" and r["predicate"] == "年龄"
   and r["via"] == "hint", "「我多少岁」语义命中", str((r["subject"], r["predicate"])))
r = dh.qrouter.route("我今年多大了")
ok(r is not None and r["predicate"] == "年龄", "换问法照样命中（泛化）")
ok(dh.qrouter.learn("年龄", "多大年纪") is None, "重复学习幂等（不重复入库）")

# ---------------------------------------------------------------- 3 L2 兜底
print("\n[3] L2：LLM 解析 → 点查 → 线索回填；幻觉谓词被闸门拒绝")
dh.store.declare_pred("年龄", 0)
dh.store.write_fact("小明", "年龄", "20岁", 1.0,
                    dh.store.add_obs(1.0, "told", "text", "其实我20岁"))
calls = {"n": 0}


def fake_llm(system, user):
    calls["n"] += 1
    return json.dumps({"subject": "我", "predicate": "年龄"}, ensure_ascii=False)


parsed, ans = llm_route_question(dh, fake_llm, "我多少岁")
ok(parsed is not None and parsed["subject"] == "小明", "解析合法：我→小明")
ok(ans is not None and not ans.abstain and "20岁" in ans.phrase(),
   "直接点查命中", ans.phrase(directness=0.5))
dh.qrouter.learn(parsed["predicate"], "我多少岁")
ok(dh.qrouter.route("我多少岁")["via"] == "hint", "回填后 L1 直达")

bad = llm_route_question(
    dh, lambda s, u: json.dumps({"subject": "我", "predicate": "飞碟数"}),
    "我有几个飞碟")
ok(bad == (None, None), "幻觉谓词（不在注册表）→ 闸门拒绝，什么都不学")
ok(calls["n"] == 1, "第二次同问法不再调 LLM（自学习生效）", "n=%d" % calls["n"])

# ---------------------------------------------------------------- 4 门槛修复
print("\n[4] 门槛修复：时间路捞到的正解不再被『混合分当余弦』误杀")
dh3 = DigitalHuman(seed=33)
o = dh3.store.add_obs(1.0, "told", "text", "李四住在海淀区")
dh3.store.write_fact("李四", "住址", "海淀区", 1.0, o)
dh3.store.commit()
# 问句认不出实体（李四不在别名词典）→ 只能靠时间路 + 真余弦门槛
ctx = collect_context(dh3, "李四的住址是海淀区吗")
facts = [(f[0], f[1], f[2]) for f in ctx["facts"]]
ok(("李四", "住址", "海淀区") in facts,
   "真余弦 ≥0.5 的近期 fact 进【事实】节", str(facts[:2]))
# 对照：无关问句不该把这条 fact 拽进来
ctx2 = collect_context(dh3, "今天股市行情怎么样")
ok(all(f[0] != "李四" for f in ctx2["facts"]), "无关问句仍被真余弦挡住")

# ---------------------------------------------------------------- 5 端到端
print("\n[5] 端到端：学过之后「我多少岁」走点查，0 次 LLM、措辞带出处")
dh4 = DigitalHuman(seed=34)
dh4.store.set_owner("小明")
dh4.store.write_fact("小明", "年龄", "20岁", 1.0,
                     dh4.store.add_obs(1.0, "told", "text", "其实我20岁",
                                       source="用户"),
                     rel=1)
dh4.qrouter.learn("年龄", "多大年纪", source="llm")
dh4.qrouter.learn("年龄", "我多少岁", source="llm")
calls2 = {"n": 0}


def fake_llm2(system, user):
    calls2["n"] += 1
    return "这个不该被调到"


out = __import__("dh.brain.graph", fromlist=["run"]).run(
    dh4, "我多少岁", llm_chat=fake_llm2)
ok(out["route"] == "fast" and out["leg"] == "fast_hit", "路由器直接进点查腿",
   "%s/%s" % (out["route"], out["leg"]))
ok("20岁" in out["reply"] and "告诉过我" in out["reply"], "来源分级措辞",
   out["reply"])
ok(calls2["n"] == 0, "LLM 一次都没调（修复前这里 3-10 秒 + 还可能答错）",
   "n=%d" % calls2["n"])

# ---------------------------------------------------------------- 6 检验回归
print("\n[6] 检验回归：亲属守卫 / 传话人剥离 / 频次问拒路由 / 错位对齐")
# 亲属称呼：「我姐姐电话」≠「我的电话」——不得消解到属主
dh6 = DigitalHuman(seed=41)
dh6.store.set_owner("小明")
dh6.store.write_fact("小明", "电话", "13800001234", 1.0,
                     dh6.store.add_obs(1.0, "told", "text", "我电话13800001234",
                                       source="用户"), rel=1)
dh6.qrouter.learn("电话", "我电话多少", source="llm")
ok(dh6.qrouter.route("我姐姐电话多少") is None,
   "亲属称呼句拒路由（宁可漏，不把姐姐的电话挂到属主头上）")
# 传话人剥离：「听张姨说，我电话是多少」的「我」是说话人不是张姨
r = dh6.qrouter.route("听张姨说，我电话是多少")
ok(r is not None and r["subject"] == "小明", "传话人前缀剥离后「我」消解到属主",
   str((r["subject"], r["predicate"])) if r else None)
# 频次/元问题拒路由（点查只会答非所问）
ok(dh6.qrouter.route("我电话号码换了多少次了") is None, "频次问拒路由")
# 错位对齐：缺向量行的线索不得让按位取值路由到错误谓词
dh7 = DigitalHuman(seed=42)
dh7.store.set_owner("小明")
with_hash = __import__("dh.memory.embed", fromlist=["make_embed_fn"]).make_embed_fn(prefer_model=False)[0]
from dh.memory.qroute import QuestionRouter                # noqa: E402
r_hash = QuestionRouter(dh7.store, with_hash, extractor=dh7.writer.ex)
r_hash.learn("年龄", "线索A", source="llm")                 # hash 口径：插表不插向量
r_hash.learn("年龄", "线索B", source="llm")                 # 同上
r_bge = dh7.qrouter                                        # bge 口径的正式路由器
r_bge.learn("住址", "住在哪里", source="llm")               # 唯一有向量行的线索
from dh.memory import embed as _em                         # noqa: E402
import numpy as _np                                        # noqa: E402
qv = _em.make_embed_fn()[0]("住在哪里")
_ids, _mat = dh7.store.load_matrix("pred_hint", model=_em.MODEL_NAME)
best = int(_np.argmax(_mat @ (qv / _np.linalg.norm(qv))))
ok(r_bge._cache[2].get(int(_ids[best])) == "住址",
   "缺向量行时按 id 映射（旧按位取值会路由到「年龄」）")

print("\n全部通过：%d 项断言" % PASS)
