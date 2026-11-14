# -*- coding: utf-8 -*-
"""主图的路由策略（大脑的决策皮层）：每个分支条件都读**记忆状态**，纯函数。

这是「记忆存取控制整个 agent」的落点：走哪条腿不由写死的流程决定，
由此刻的记忆与心智状态决定——

    输入（全部是记忆/心智状态，无副作用）：
        parsed     抽取层从原话里认出的三元组（extract -> 记忆的形状）
        ctx        检索上下文（collect_context 一次收集的 A/B/C 三路原料）
        disp       行为倾向（PAD -> disposition，情绪第一次改控制流）
        ack        自报家门的确认语（属主登记 -> alias/meta）
        llm_chat   配没配 Key（决定 RAG 这条腿存不存在）

    输出：route ∈
        ack        自报家门确认——登记属主是一句话里唯一确定要做的事
        defer      敷衍——情绪说不动 + 这句话本身没内容：不调 LLM，应一声就过
        fast       事实问句点查优先（µs 级、0 token；点查弃权由 compose 降级）
        offline    未配 Key——记忆点查照常 + 离线兜底
        llm        其余一切 → RAG（三路检索 + LLM 组织语言）

顺序与旧 dh/turn.py 的 node_plan 逐条对齐（行为不变），三条实测依据随行：
    * defer 只在配了 Key 时生效——离线模式没有「硬凑一段话」的问题，
      兜底语本身就是短的（旧 plan 的判断顺序如此）。
    * fast 只看句形、不管有没有 Key——离线模式下事实问句照样点查。
    * 「什么都没检索到」不是 defer 的判据：recall 返回 z 标准化分数，
      候选池有条目就必然填满 k_gate（实测「嗯」也拿到 hits=7），
      所以判据只能是「这句话值不值得回应」。
"""
import re

from ..limbic.disposition import disposition   # 右脑的调制面（唯一情绪来源）
from ..core.tick import looks_like_question

_PUNCT = "。！？；，,.!?;：:"

# 敷衍用语：故意短、不承接、不追问（低唤醒/持续低落时人就是这么应的）
DEFER_LINES = ("嗯。", "哦。", "嗯，知道了。")

_ROUTES = ("ack", "defer", "fast", "offline", "llm")


def is_filler(text, parsed, ctx=None):
    """这句话值不值得回。三个条件同时成立才算「不值得」：

    ① 解析不出结构（parsed 为 None）
    ② 认不出任何实体（问「嗯」和问「我姐在哪」的区别就在这里）
    ③ 短：去掉标点后不超过 4 个字

    只有 ①②③ 都成立才敷衍。少了 ② 会把「我姐在哪」这种认不出别名但
    明确在问事的话也敷衍掉；少了 ③ 会把「今天好累啊」这种带情绪的倾诉
    也敷衍掉（实测 5 字就够长了，那是该接的话）。
    """
    if parsed is not None:
        return False
    if ctx and ctx.get("ents"):
        return False
    body = "".join(ch for ch in (text or "") if ch not in _PUNCT)
    return len(body) <= 4


def current_disposition(dh):
    """此刻的行为倾向快照（2 µs，不缓存——缓存要在每次写情绪时失效）。"""
    return disposition(dh)


def defer_line(dh, disp=None):
    """敷衍用语：低耐心时轮换，其余一律「嗯。」（与旧 compose 分支一致）。"""
    disp = disp or current_disposition(dh)
    return (DEFER_LINES[int(dh.affect.t_ms) % len(DEFER_LINES)]
            if disp.patience < 0.2 else "嗯。")


def decide(dh, st):
    """主图的分支函数：读记忆状态 -> (route, reason)。

    st 是一轮对话的状态 dict（dh/brain/nodes.py 的 TurnState）。
    除路由名外，把两个派生量写回 st 供 compose 消费（compose 不重算）：
        try_fast  句形是否像事实问句（offline 腿的点查兜底也读它）
        disp      此刻的行为倾向快照
    """
    if st.get("ack") is not None:
        return "ack", "自报家门：登记属主是一句话里唯一确定要做的事"
    disp = (st.get("ctx") or {}).get("disp") or current_disposition(dh)
    parsed = st.get("parsed")
    try_fast = bool(parsed is not None
                    and looks_like_question(st["text"])
                    and st.get("vision_desc") is None)
    st["try_fast"] = try_fast
    st["disp"] = disp
    if st.get("llm_chat") is None:
        return "offline", "未配 API Key：记忆点查 + 离线兜底"
    if disp.mode == "defer" and is_filler(st["text"], parsed, st.get("ctx")):
        return "defer", "倾向 defer（%s）+ 无结构短句：情绪决定不调 LLM" % disp.label
    if try_fast:
        return "fast", "事实问句：点查优先（µs 级），弃权才降级 RAG"
    return "llm", "开放问题：三路检索 + LLM（RAG）"


def should_defer(dh, text, parsed, llm_configured, ctx=None):
    """给网页端的单点闸门（web/on_chat 没有完整 st，只问这一个决定）。

    与 decide 的 defer 分支同一份判据：配了 Key + 说不动 + 这句话没内容。
    ctx 给不出时（on_chat 没有 collect_context），is_filler 的第②条
    （认得出实体就不敷衍）在这里用别名词典补齐——「老王呢」这种
    认得出实体的短句不能被敷衍（实测对齐 CLI 行为）。"""
    if not llm_configured:
        return False
    disp = (ctx or {}).get("disp") or current_disposition(dh)
    if disp.mode != "defer":
        return False
    if ctx is None:
        ctx = {"ents": [e[2] for e in dh.ex.find_entities(text or "")]}
    return is_filler(text, parsed, ctx)
