# -*- coding: utf-8 -*-
"""主图：节点表 + 条件边 + 执行器。这是控制整个 agent 的大脑。

图的形状（plan 节点内嵌 policy.decide 分支——「条件边」由记忆状态算出）：

    START
      │ begin()：只跑这两步（~200 µs，听见即落盘——打断安全的地基）
      ▼
    guard ──► perceive ──┐
                         │ run(resume=st) 从这里续跑
      ┌──────────────────┘
      ▼
    recall ──► plan ══[policy.decide：ack/defer/fast/offline/llm]══► compose
                                                                   │
                                        affect ◄───────────────────┘
                                          │        （出锁的 compose 回来先验 epoch）
                                       finish ──► speak ──► END

锁的边界由节点表第二列决定，runner 只在边界处 acquire/release；
compose 是唯一出锁节点（LLM 那 3-10 秒不持锁，用户随时插话）。

epoch 中断协议（外部递增整数，CLI 输入泵与 web on_chat 各自维护）：
    compose 出锁前记住它，回来先比对；不一致就丢弃结果、跳过 affect/finish
    ——用户的话已在 perceive 落盘，丢的只是「小星的回答」。

begin/run 拆成两次调用是给双线程用的：
    输入泵： st = begin(dh, text, ...)     → 立刻 commit，回去读下一句
    worker： run(dh, text, ..., resume=st) → 出锁计算，回来验 epoch
"""
import time

from . import nodes as _nodes
from .nodes import (_NODE_FN, _ensure_heard, _trust)   # noqa: F401  再导出

# 节点表：(名字, 是否需要持锁)。顺序即主路执行顺序。
GRAPH = (
    ("guard", True),
    ("perceive", True),
    ("recall", True),
    ("plan", True),
    ("compose", False),        # ← 唯一出锁节点
    ("affect", True),
    ("finish", True),
    ("speak", False),
)

_ROUTE_DESC = ("plan ═[policy.decide：ack | defer | fast | offline | llm]═ "
               "► compose（fast/llm 弃权降级链在同节点内）")

# 右脑（dh/limbic）对每一步的调制面：大脑只从这里拿情绪，不读 PAD 裸值
_LIMBIC_FACE = {
    "guard": "",
    "perceive": "右脑：valence 评价入口 + gate→obs.salience 编码门（记得多深）",
    "recall": "右脑：mood（情绪一致检索）+ breadth（k_gate 缩放）",
    "plan": "右脑：talkativeness（defer/terse 腿路）+ patience（prompt_line 约束）",
    "compose": "右脑：directness（点查措辞）+ patience（defer 用语轮换）",
    "affect": "右脑本体：<appraisal> → appraise_to_pad → OU 驱动注入",
    "finish": "",
    "speak": "右脑：四出口（语音韵律 rate/pitch/energy + tone）",
}


def describe():
    """主图的可读形状（调试/文档用）。图是数据，不是藏在 if 链里的流程；
    每个节点标注右脑的调制面——情绪控制每一步，一眼可见。"""
    lines = ["START", "  │ begin(): guard → perceive（听见即 commit）", "  ▼"]
    for name, need_lock in GRAPH[2:]:
        lock = "🔒" if need_lock else "🔓"
        face = _LIMBIC_FACE.get(name, "")
        lines.append("%s %s%s" % (lock, name,
                                  ("　─ %s" % face) if face else ""))
        if name == "plan":
            lines.append("  ║ %s" % _ROUTE_DESC)
        elif name != GRAPH[-1][0]:
            lines.append("  │")
    lines.append("END")
    return "\n".join(lines)


# ---------------------------------------------------------------- 执行器
def begin(dh, text, llm_chat=None, vision_desc=None, epoch=None):
    """只跑到 perceive 为止：听见 + 立刻落盘。~200 µs。

    **打断安全靠这一步**：用户在 worker 正算 LLM 的时候又说了句话，输入泵
    必须能马上把这句话记下来——不能等上一轮 compose 回来。所以把
    「听见」和「回答」拆成两次调用：
        输入泵： st = begin(dh, text, ...)   → 立刻 commit，回到 input()
        worker： run(dh, text, ..., resume=st) → 出锁计算，回来验 epoch

    begin 自己拿锁、自己放锁，不留给调用方——它必须是原子的。
    """
    st = dict(text=text, llm_chat=llm_chat, vision_desc=vision_desc,
              epoch=epoch, t=time.time())
    with dh.store.lock:
        _NODE_FN["guard"](dh, st)
        _NODE_FN["perceive"](dh, st)
    return st


def run(dh, text, llm_chat=None, vision_desc=None, epoch=None,
        epoch_now=None, on_stale=None, resume=None):
    """跑一轮。返回 dict，字段与旧 dh/turn.run 完全一致（调用方不用改），
    另加 route / route_why 两个观测字段（主图这轮走了哪条腿、为什么）。

    epoch / epoch_now：外部打断协议。出锁的 compose 回来若发现
    `epoch != epoch_now()`，结果作废——跳过 affect/finish，只留下已经
    commit 过的「用户原话」。CLI 的输入泵和 web 的 on_chat 共用同一套。

    resume：已由 begin() 跑过 guard+perceive 的状态。给了就跳过这两个节点
    （否则同一句话会写两条 obs —— 实测踩过）。
    """
    if resume is not None:
        st = resume
        # epoch 以 begin 存进 st 的为准——resume 时忘传 epoch 参数不再
        # 静默失去打断保护（main.py 传的是 st["epoch"，契约统一到 st）
        if epoch is None:
            epoch = st.get("epoch")
        st.setdefault("epoch", epoch)
        st.setdefault("llm_chat", llm_chat)
        st.setdefault("vision_desc", vision_desc)
        skip = ("guard", "perceive")
    else:
        st = dict(text=text, llm_chat=llm_chat, vision_desc=vision_desc,
                  epoch=epoch, t=time.time())
        skip = ()
    locked = False
    walked = []
    try:
        for name, need_lock in GRAPH:
            if name in skip:
                continue
            if need_lock and not locked:
                dh.store.lock.acquire()
                locked = True
            elif not need_lock and locked:
                dh.store.lock.release()
                locked = False
            # 出锁节点回来先验 epoch：作废则跳过所有会写库的后续节点
            if (name in ("affect", "finish") and epoch is not None
                    and epoch_now is not None and epoch != epoch_now()):
                st["_stale"] = True
                if on_stale is not None:
                    on_stale(st)
                break
            _NODE_FN[name](dh, st)
            walked.append(name)
    finally:
        if locked:
            dh.store.lock.release()
    if st.get("_stale"):
        return dict(reply="", mood=_nodes._mood(dh), prosody={}, flags=[],
                    sources=[], used_memory=False, stale=True,
                    hear_out=st.get("hear_out") or {}, cited=[], dropped=True,
                    route=st.get("route", ""), route_why="", deferred=False,
                    nodes=walked)
    return dict(reply=st.get("reply", ""),
                mood=st.get("mood") or _nodes._mood(dh),
                prosody=st.get("prosody") or {},
                flags=st.get("flags") or [],
                sources=st.get("sources") or [],
                used_memory=bool(st.get("used_memory")),
                stale=bool(st.get("stale")),
                hear_out=st.get("hear_out") or {},
                cited=st.get("cited") or [],
                trust=(st.get("hear_out") or {}).get("trust"),
                ctx=st.get("ctx") or {},
                route=st.get("route", ""),
                route_why=st.get("route_why", ""),
                leg=st.get("leg", ""),
                deferred=bool(st.get("deferred")),
                nodes=walked)
