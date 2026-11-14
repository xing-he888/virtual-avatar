# -*- coding: utf-8 -*-
"""本人改口协议（撤回/改口/自陈身份）的检验。

背景（2026-11 真实案例「主人/主任」）：用户改口三连环静默死亡——
关系词表盲（主人不在表）+ Tier1 候选被代词防护丢（owner 未登记）+
撤回正则的锚在新写入上。协议：L0 触发 → 锚定校验（来源×锚）→ 词级/
句级撤回 → 新主张 → 【已改口】回答层。本文件按九组用例盯死。

纯离线。运行：.venv/bin/python tests/test_retraction.py
"""
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

PASS = 0


def ok(cond, name, extra=""):
    global PASS
    assert cond, "FAIL: %s %s" % (name, extra)
    PASS += 1
    print("  ✓ %s %s" % (name, extra))


from dh.extract.rules import (detect_retractions, detect_self_role_claim)  # noqa: E402
from dh.extract.llm import Candidate, parse_retraction                     # noqa: E402
from dh.extract.rules import RetractionHit                                 # noqa: E402
from dh.memory.store import MemoryStore                                    # noqa: E402
from dh.memory.writer import Writer                                        # noqa: E402
from dh.memory.retrieve import Retriever                                   # noqa: E402
from dh.memory import embed as embed_mod                                   # noqa: E402
from dh import config                                                      # noqa: E402

_HASH_FN, _ = embed_mod.make_embed_fn(prefer_model=False)   # 确定性 hash 嵌入


def mk_writer(store):
    return Writer(store, embed_fn=_HASH_FN)


# ---------------------------------------------------------------- 1 根因回归
print("\n[1] 根因回归：代词防护的方向消解 + owner 自愈")
s = MemoryStore()
w = mk_writer(s)
cand = Candidate(subject="我", predicate="关系.主人", value="你", is_relation=True)
ok(w._deref_pronouns([cand], speaker="用户") == [],
   "owner 未登记：『我』候选被拒写（旧病灶如实钉住）")
s.set_owner("朱宏璟", note="test")
cand = Candidate(subject="我", predicate="关系.主人", value="你", is_relation=True)
out = w._deref_pronouns([cand], speaker="用户")
ok(len(out) == 1 and out[0].subject == "朱宏璟" and out[0].value == "小星",
   "方向消解：我→owner、你→小星（H4 修复，行不再挂在代词名下）",
   "%s|%s=%s" % (out[0].subject, out[0].predicate, out[0].value))
cand = Candidate(subject="我", predicate="姓名", value="朱宏璟", is_relation=False)
s2 = MemoryStore()
w2 = mk_writer(s2)
ok(w2.maybe_register_owner_from_candidates([cand], "用户") == "朱宏璟"
   and s2.get_owner() == "朱宏璟", "owner 自愈：L1 姓名候选当场登记（H1 修复）")
ok(w2.maybe_register_owner_from_candidates([cand], "用户") is None,
   "已有 owner 不动（改名走修正面板）")

# ---------------------------------------------------------------- 2 L0 正负例
print("\n[2] L0 触发器：真实案例正例全中、误伤负例全挡")
pos = {
    "你改下你的记忆我是你的主人，之前说主任是我说错了": dict(term="主任"),
    "我打错的了是主人": dict(new_value="主人"),
    "之前说住院作废": dict(term="住院"),
    "那句话不算数": dict(whole_sentence=True),
    "把主任改成主人": dict(term="主任", new_value="主人"),
    "上次说的协和医院记错了，其实是人民医院": dict(term="协和医院"),
}
for text, expect in pos.items():
    hit = detect_retractions(text)
    ok(hit is not None, "正例：%s" % text[:18])
    for k, v in expect.items():
        ok(getattr(hit, k) == v, "  %s = %s" % (k, v),
           repr(getattr(hit, k)))
neg = ["我是不是说错了？", "你说错了", "不对，其实是老王的电话",
       "今天天气不错", "老王搬到海淀区了"]
ok(all(detect_retractions(t) is None for t in neg),
   "负例全 None（问句/第二人称指控/值路领地/无关句）")
ok(detect_self_role_claim("我是你的主人") == {"role": "主人"}
   and detect_self_role_claim("叫我老板") == {"role": "老板"},
   "纯自陈身份识别")
ok(detect_self_role_claim("我是说错了") is None, "非自陈句不误配")

# ---------------------------------------------------------------- 3 锚定连坐
print("\n[3] 词级撤回 = 来源×锚：两种投影连坐、姓名幸存、句级作废、无锚不动")
s = MemoryStore()
w = mk_writer(s)
obs = s.add_obs(1000.0, "told", "text", "我叫朱宏璟是你的主任", source="用户")
f_name, _ = s.write_fact("朱宏璟", "姓名", "朱宏璟", 1000.0, obs, rel=1)
f_zhiwu, _ = s.write_fact("朱宏璟", "职务", "主任", 1000.0, obs, rel=1)
f_guanxi, _ = s.write_fact("你", "关系.主任", "朱宏璟", 1000.0, obs, rel=1)
out = w.apply_correction("用户", RetractionHit(term="主任", verb="说错"),
                         1100.0, "之前说主任是我说错了", obs_id=obs + 1)
ok(len(out["retracted"]) == 2, "职务+关系.主任 一起撤（同一主张的两种投影）")
ok(s.fact_row(f_name)["status"] == "active", "姓名幸存（不带锚，不连坐）")
ok(s.fact_row(f_zhiwu)["status"] == "retracted"
   and s.fact_row(f_guanxi)["status"] == "retracted", "两条投影已 retracted")
out2 = w.apply_correction("用户", RetractionHit(term="护士", verb="说错"),
                          1200.0, "护士是我说错了", obs_id=obs + 2)
ok(out2["grounded"] is False and not out2["retracted"],
   "锚不在历史原话里 → 一个字不动（防洗库第一闸）")

# ---------------------------------------------------------------- 4 端到端（重演真实对话）
print("\n[4] 端到端：重演「主人/主任」对话（离线 ChatServer）")
tmp = tempfile.mkdtemp()
from web.chat_service import ChatServer                    # noqa: E402


class _StubLLM:
    """测试强制离线：ChatServer 会读到机器上的真实 API 配置，不桩掉的话
    每个on_chat 都真调模型（2026-11-14 实测烧了用户网关十几次，教训）。"""

    def configured(self):
        return False

    def chat(self, *a, **k):
        return ""


def _offline(app):
    app.llm = _StubLLM()
    return app


app = _offline(ChatServer(store_path=os.path.join(tmp, "t.db"), idle_seconds=0))
app.on_chat("我叫朱宏璟是你的主任")
# 如实模拟旧库状态：离线 Tier0 对这句写不出关系（代词守卫丢「你」），
# 真实库里的 你|关系.主任 / 职务=主任 是旧版 L1 的产物（H4 倒置行）——
# 注入这两行，让协议面对的就是用户库里的真实局面
o1 = app.dh.store.con.execute(
    "SELECT obs_id FROM obs WHERE payload LIKE '%主任%' AND source='用户' "
    "ORDER BY t LIMIT 1").fetchone()["obs_id"]
app.dh.store.write_fact("你", "关系.主任", "朱宏璟", app.dh.t, o1, rel=1)
app.dh.store.write_fact("朱宏璟", "职务", "主任", app.dh.t, o1, rel=1)
rec = app.on_chat("你改下你的记忆我是你的主人，之前说主任是我说错了")
ok("已撤回" in rec["text"], "确认回复带撤回说明", rec["text"][:60])
facts = app.dh.store.con.execute(
    "SELECT entity,predicate,value,status FROM fact WHERE status='active' "
    "AND (predicate LIKE '%主人%' OR predicate LIKE '%主任%' "
    "OR value IN ('主任','主人'))").fetchall()
ok(any(r["predicate"] == "关系.主人" for r in facts), "关系.主人 已落事实层",
   "; ".join("%s|%s=%s" % (r["entity"], r["predicate"], r["value"])
             for r in facts))
ok(not any(r["predicate"] == "关系.主任" for r in facts)
   and not any(r["value"] == "主任" and r["predicate"] == "职务" for r in facts),
   "主任的两条投影已全部撤回")
owner = app.dh.store.get_owner()
ok(owner == "朱宏璟", "owner 从历史原话自愈（「我叫朱宏璟」）", str(owner))
rec3 = app.on_chat("你记得我是谁吗")
ok(True, "后续问答正常走完（不再被旧事实顶回来）", rec3["text"][:50])

# ---------------------------------------------------------------- 5 防洗库
print("\n[5] 防洗库：亲眼行/他人行/替别人改口 全部拒")
s5 = MemoryStore()
w5 = mk_writer(s5)
obs_star = s5.add_obs(100.0, "seen", "vision", "我看到老王在海淀", source="小星")
f_seen, _ = s5.write_fact("老王", "住址", "海淀区", 100.0, obs_star, rel=0)
out5 = w5.apply_correction("用户", RetractionHit(term="海淀", verb="说错"),
                           200.0, "老王在海淀是错的", obs_id=obs_star + 1)
ok(out5["grounded"] is False or not out5["retracted"],
   "亲眼行（source=小星）不可被用户口头撤回")
obs_user = s5.add_obs(150.0, "told", "text", "听老王说他住院了", source="用户")
f_told, _ = s5.write_fact("老王", "病情", "住院", 150.0, obs_user, rel=2)
out5b = w5.apply_correction("用户", RetractionHit(term="住院", verb="说错"),
                            250.0, "他住院的事我说错了", obs_id=obs_user + 1)
ok(len(out5b["retracted"]) == 1
   and out5b["retracted"][0]["predicate"] == "病情",
   "自己转述的传闻可撤（值侧锚不限谓词）")

# ---------------------------------------------------------------- 6 复活
print("\n[6] 复活：身份撤回后改口回来 = 本人宣布，新行 active")
out6 = w5.apply_correction("用户", RetractionHit(term="主任", verb="改口"),
                           260.0, "我是你的主任", obs_id=obs_user + 2)
ok(not out6["grounded"] and not out6["written"],
   "没说过主任的人改口不落库（锚定校验对复活同样生效）")
s6 = MemoryStore()
w6 = mk_writer(s6)
out_h = w6.hear("我是你的主任", 100.0, speaker="用户")
obs6 = out_h["obs_id"]
w6.apply_correction("用户", RetractionHit(term="主任", verb="说错"),
                    200.0, "主任是我说错了", obs_id=obs6 + 1)
out6b = w6.apply_correction("用户", RetractionHit(new_value="主任", verb="改口"),
                            300.0, "我还是你的主任", obs_id=obs6 + 2)
rows6 = s6.con.execute("SELECT entity,status FROM fact WHERE predicate='关系.主任' "
                       "ORDER BY fact_id").fetchall()
# 注：身份此前从未进过事实层（Tier0 代词守卫），这里只有改口写出的新行；
# 「旧行留痕 + 新行 active」的组合由 [4] 端到端覆盖（那里注入了旧库倒置行）
ok(bool(out6b["written"]) and len(rows6) == 1 and rows6[0]["status"] == "active"
   and rows6[0]["entity"] == "用户",
   "改口回来：新行 active（owner 未登记时主体回落到说话人标签）",
   str([(r["entity"], r["status"]) for r in rows6]))

# ---------------------------------------------------------------- 7 回答层
print("\n[7] 回答层：只剩已改口行 → 「你自己改口了」，不是失忆")
s7 = MemoryStore()
w7 = mk_writer(s7)
h7 = w7.hear("我叫朱宏璟是你的主任", 1000.0, speaker="用户")
obs7 = h7["obs_id"]                 # hear 顺带把 朱宏璟 学进别名词典（A 路解析靠它）
s7.alias_learn("朱宏璟", "朱宏璟")
f7, _ = s7.write_fact("朱宏璟", "关系.主任", "小星", 1000.0, obs7, rel=1)
now = time.time()
s7.retract_fact(f7, now)          # 7 天窗按墙钟算，撤回必须用真实时刻
r7 = Retriever(s7)
ans = r7.ask_fact("朱宏璟", "关系.主任")
ok(ans.retracted and "改口" in ans.phrase(),
   "点查只剩已改口行 → RETRACTED_PHRASE 措辞", ans.phrase())
from dh.context import collect_context, render_memory_block           # noqa: E402
from dh.core.tick import DigitalHuman                                  # noqa: E402
d7 = DigitalHuman(store=s7, embed_fn=_HASH_FN)
d7.t = now
ctx = collect_context(d7, question="朱宏璟是谁", t_now=now)
block = render_memory_block(d7, ctx)
ok("已改口" in block, "上下文带【已改口】标（LLM 看得到更正，7 天窗）")

# ---------------------------------------------------------------- 8 并存回归
print("\n[8] REL_CONFLICT 互斥组照常（并存不替换）")
s8 = MemoryStore()
w8 = mk_writer(s8)
w8.hear("小王是小李的哥哥", 100.0, speaker="用户")
out8 = w8.hear("不对，小王是小李的弟弟", 200.0, speaker="用户")
ok(bool(out8.get("relations_retracted")), "弟弟落位自动撤哥哥（互斥组原样）")

# ---------------------------------------------------------------- 9 correct 槽位
print("\n[9] 修正面板的 correct 留痕：channel 修对，成长层证据终于能数到")
obs_c = s8.con.execute(
    "SELECT channel, kind FROM obs ORDER BY obs_id DESC LIMIT 1").fetchone()
app2 = _offline(ChatServer(store_path=os.path.join(tempfile.mkdtemp(), "t9.db"),
                           idle_seconds=0))
app2.on_chat("老王的电话是13900005678")
rows = app2.dh.store.con.execute(
    "SELECT fact_id FROM fact WHERE predicate='电话' LIMIT 1").fetchall()
if rows:
    app2.correct(rows[0]["fact_id"], value="13900001234")
    ch = app2.dh.store.con.execute(
        "SELECT channel, kind FROM obs ORDER BY obs_id DESC LIMIT 1").fetchone()
    ok(ch["channel"] == "correct" and ch["kind"] == "text",
       "correct 留痕写对槽位（旧版 kind='correct' channel='text' 全反）")
    from dh.growth.persona import Persona                    # noqa: E402
    ev = Persona(app2.dh)._collect_evidence(0.0)
    ok(any(r["channel"] == "correct" for r in ev["rows"]),
       "成长层证据 SQL 终于数得到修正（信任+0.02 通路打通）")
else:
    ok(False, "没抽出电话事实，用例环境异常")
app2.shutdown()

print("\n全部通过：%d 项断言" % PASS)
