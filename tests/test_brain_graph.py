# -*- coding: utf-8 -*-
"""主图（dh/brain）的检验：五条路由腿 / epoch 打断 / begin-resume 防重 / 兼容 shim。

纯离线（llm_chat 用 Mock，不调任何 API）。运行：
    .venv/bin/python tests/test_brain_graph.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from dh.limbic.emotion import PAD                        # noqa: E402
from dh.core.tick import DigitalHuman                    # noqa: E402
from dh.brain import begin, run, describe                # noqa: E402
from dh.brain import policy                              # noqa: E402

PASS = 0


def ok(cond, name, extra=""):
    global PASS
    assert cond, "FAIL: %s %s" % (name, extra)
    PASS += 1
    print("  ✓ %s %s" % (name, extra))


def fresh(seed=101):
    return DigitalHuman(seed=seed)


def mock_llm(calls, reply="好呀。<appraisal>{\"desirability\":0.6,"
                         "\"agency\":\"circumstance\",\"novelty\":0.1,"
                         "\"goal_relevance\":0.5}</appraisal>"):
    def _chat(prompt, text):
        calls.append((prompt, text))
        return reply
    return _chat


# ---------------------------------------------------------------- 1 fast 腿
print("\n[1] fast：事实问句点查优先，0 次 LLM 调用")
dh = fresh()
dh.hear("老王的电话是 13800001234", speaker="张姨", t=10.0)
calls = []
out = run(dh, "老王现在电话是多少", llm_chat=mock_llm(calls))
ok(out["route"] == "fast", "route=fast", out["route_why"])
ok("13800001234" in out["reply"], "点查命中给号码", out["reply"][:30])
ok(len(calls) == 0, "LLM 一次都没调", "n=%d" % len(calls))
ok(out["used_memory"] and not out["stale"], "used_memory")

# ---------------------------------------------------------------- 2 offline 腿
print("\n[2] offline：没配 Key 也工作，事实问句照常点查")
dh = fresh()
dh.hear("老王的电话是 13800001234", speaker="张姨", t=10.0)
out = run(dh, "老王现在电话是多少", llm_chat=None)
ok(out["route"] == "offline", "route=offline", out["route_why"])
ok("13800001234" in out["reply"], "离线模式点查照样命中", out["reply"][:40])
out = run(dh, "今天天气怎么样", llm_chat=None)
ok(out["route"] == "offline" and out["reply"], "非问句走离线兜底")

# ---------------------------------------------------------------- 3 defer 腿
print("\n[3] defer：情绪决定不调 LLM（低唤醒 + 应答词）")
dh = fresh()
dh.affect.state = PAD(0.0, -0.9, 0.0)          # 发困态：talkativeness≈0.10
calls = []
out = run(dh, "嗯", llm_chat=mock_llm(calls))
ok(out["route"] == "defer", "route=defer", out["route_why"])
ok(out["reply"] in policy.DEFER_LINES or out["reply"] == "嗯。",
   "敷衍用语", out["reply"])
ok(out.get("deferred") is True, "deferred 标记")
ok(len(calls) == 0, "LLM 一次都没调", "n=%d" % len(calls))
out = run(dh, "今天好累啊", llm_chat=mock_llm(calls))
ok(out["route"] == "llm", "带情绪的倾诉不敷衍（is_filler 拦住）",
   out["route_why"])

# ---------------------------------------------------------------- 4 llm 腿（RAG）
print("\n[4] llm：开放问题走 RAG，<appraisal> 落情绪")
dh = fresh()
p0 = dh.affect.state.p
calls = []
out = run(dh, "你觉得生活有什么意义", llm_chat=mock_llm(calls))
ok(out["route"] == "llm", "route=llm", out["route_why"])
ok(out["leg"] == "llm", "leg=llm", out["leg"])
ok(len(calls) == 1, "LLM 恰好一次", "n=%d" % len(calls))
ok(out["reply"] == "好呀。", "<appraisal> 标签被剥离", out["reply"])
ok(dh.affect.state.p > p0, "评价落情绪：P 抬升",
   "%+.2f→%+.2f" % (p0, dh.affect.state.p))

print("\n[4b] fast 弃权降级：问句形但库里没有 → 点查弃权掉进 RAG")
dh = fresh()
calls = []
out = run(dh, "你觉得老王这个人怎么样", llm_chat=mock_llm(calls))
ok(out["route"] == "fast", "计划腿 fast（问句启发式判成 病情 问句）",
   out["route_why"])
ok(out["leg"] == "fast_fallback_llm", "实际腿：弃权降级 LLM", out["leg"])
ok(len(calls) == 1 and out["reply"] == "好呀。", "降级后由 LLM 生成")

# ---------------------------------------------------------------- 5 ack 腿
print("\n[5] ack：自报家门登记属主")
dh = fresh()
out = run(dh, "我叫王小明", llm_chat=None)
ok(out["route"] == "ack", "route=ack", out["route_why"])
ok("王小明" in out["reply"], "确认语带名字", out["reply"])
ok(dh.store.get_owner() == "王小明", "属主已登记")

# ---------------------------------------------------------------- 6 epoch 打断
print("\n[6] epoch：出锁回来发现过期 → 丢弃回答，用户原话保住")
dh = fresh()
st = begin(dh, "讲个故事吧", llm_chat=mock_llm([]), epoch=5)
out = run(dh, "讲个故事吧", llm_chat=mock_llm([]), epoch=5,
          epoch_now=lambda: 6, resume=st)
ok(out.get("dropped") is True, "dropped 标记")
ok(out["reply"] == "", "回答被丢弃")
ok(out["hear_out"].get("obs_ids"), "用户原话已在 perceive 落盘")

# ---------------------------------------------------------------- 7 begin/resume 防重
print("\n[7] begin+resume：同一句话只写一条 obs（实测踩过的坑）")
dh = fresh()
st = begin(dh, "老王住在海淀区", llm_chat=None, epoch=1)
n_begin = dh.store.obs_count()
out = run(dh, "老王住在海淀区", llm_chat=None, resume=st)
n_after = dh.store.obs_count()
ok(n_after - n_begin == 1, "resume 跳过 perceive，只加小星回复一条 obs",
   "+%d" % (n_after - n_begin))
texts = [r["payload"] for r in dh.store.con.execute(
    "SELECT payload FROM obs WHERE source='用户'")]
ok(texts.count("老王住在海淀区") == 1, "用户原话恰好一条（begin 写的，run 不重写）")

# ---------------------------------------------------------------- 8 图是数据
print("\n[8] describe()：主图可打印")
d = describe()
for name in ("guard", "perceive", "recall", "plan", "compose",
             "affect", "finish", "speak"):
    ok(name in d, "节点 %s 在图中" % name)

# ---------------------------------------------------------------- 9 兼容 shim
print("\n[9] dh.turn shim：旧导入路径原样可用")
from dh.turn import run as t_run, begin as t_begin, _trust   # noqa: E401
ok(t_run is run and t_begin is begin, "begin/run 同一对象")
ok(callable(_trust), "_trust 可导入")
out = run(fresh(), "老王住在海淀区", llm_chat=None)
for key in ("reply", "mood", "prosody", "flags", "sources", "used_memory",
            "stale", "hear_out", "cited", "nodes"):
    ok(key in out, "返回字段兼容：%s" % key)

print("\n全部通过：%d 项断言" % PASS)
