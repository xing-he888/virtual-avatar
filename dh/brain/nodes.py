# -*- coding: utf-8 -*-
"""主图的节点实现（从 dh/turn.py 迁入）：每个节点做一件事，路由交给 policy。

节点之间的顺序是数据依赖，不是排版：

    guard      锁定   自报家门 / 问句解析（parse_question 唯一一次）
    perceive   锁定   dh.hear + 问句留底 + 可信度  →  **立刻 commit**
    recall     锁定   三路检索一次（collect_context）
    plan       锁定   policy.decide 定路由 + 组装 prompt
    compose    出锁   ack/defer/fast/llm/offline 五条腿三选一执行 ← 唯一秒级节点
    affect     锁定   <appraisal> 落情绪
    finish     锁定   自己的话入记忆 + 边界提交 →  commit
    speak      出锁   TTS 派发（丢线程）

与旧 turn.py 的差异只有两处，行为逐条对齐：
    * plan 不再用 if 链硬编码走向，改为调用 brain/policy.decide——
      「记忆存取控制 agent」的决策集中在一份可单测的纯函数里；
    * compose 按 st["route"] 分派（ack/defer 两腿与 try_fast/llm/offline
      的降级链与旧逻辑一字不差）。
"""
import json
import time

from . import policy
from .. import config
from .. import llm_bridge
from ..context import collect_context, render_memory_block
from ..core.tick import looks_like_question

# ---------------------------------------------------------------- 小工具
def _ensure_heard(store, hear_out, text, speaker, t):
    """问句不断言（tick 跳过 writer），但经历层的原话永远要留底。

    聊天记录重建、时光回溯、补记锚都靠它。没有这一步，CLI 里用户的问句
    一条都不进库——网页端有 `_ensure_heard` 兜着，CLI 从来没兜过。
    """
    if not (hear_out.get("obs_ids") or []):
        oid = store.add_obs(t, "told", "text", text, source=speaker,
                            rel=1, seg_bnd=0.0)
        hear_out["obs_ids"] = [oid]
    return hear_out


_PUNCT = policy._PUNCT
# 值后面允许跟着的收尾字：语气词和「了」。除此之外还剩字就是被截断了。
_TAIL_OK = _PUNCT + "了啦吧呢啊哦呀么的啦"


def _trust(text, parsed, dh):
    """正则这次抽得可不可信：trusted / suspect / empty。

    只为两件已实测的失败服务（`LateExtractor` 的启动判据由它决定）：
        1  主语记错人：`张姨说老王出院了` → subject=张姨（应是老王）。
           判据：句中实体 ≥2 且 subject 不是最后提到的那个。
        2  值被词典截断：`老王的电话是协和医院心内科` → value=协和医院。
           判据：value 之后还跟着同句的非标点字符。
    """
    if parsed is None:
        # parse_question 只认问句；陈述句它是 None，但 parse() 能抽出三元组。
        # 不试这一步，「老王住院了」这种正则明明抽对了的句子也会被判 empty
        # 而白跑一次 LLM 补记。
        parsed = dh.ex.parse(text)
        if parsed is None:
            return "empty"
    val = parsed.get("value") or ""
    if not val:
        # parse_question 对陈述句也会返回真值 dict，但 value 是空的
        # （`老王的电话是13800001234` 就是这种）。别据此判 suspect——
        # 再走一次陈述解析，真抽出来了就算数。
        alt = dh.ex.parse(text)
        if alt is not None and alt.get("value"):
            return _trust(text, alt, dh)
        if parsed.get("predicate"):
            # 谓词认出来了、值终究是空：`老王的电话是协和医院心内科`
            # 就是这种——值被词典查丢了。写不出事实，交给 LLM。
            return "suspect"
        return "trusted"
    i = text.find(val)
    if i >= 0:
        tail = text[i + len(val):].strip()
        if tail and tail[0] not in _TAIL_OK:
            return "suspect"
    ents = dh.ex.find_entities(parsed.get("body") or text)
    if len(ents) >= 2 and ents[-1][2] != parsed.get("subject"):
        return "suspect"
    return "trusted"


def _mood(dh):
    st = dh.affect.state
    return dict(p=round(st.p, 3), a=round(st.a, 3), d=round(st.d, 3))


def _disp(dh):
    """当前行为倾向。2 µs，不缓存——缓存要在每次写情绪时失效，不值。"""
    return policy.current_disposition(dh)


# ---------------------------------------------------------------- 节点
def node_guard(dh, st):
    """自报家门 + 问句解析。问句路由全流程只此一次（属主消解/线索/语义）。"""
    st["t"] = time.time()
    st["ack"] = llm_bridge.register_owner_if_intro(dh, st["text"])
    st["parsed"] = dh.qrouter.route(st["text"])
    # 会走 LLM 的话先听成中性——情绪改由 LLM 按性格的评价驱动；
    # 词典只在离线/快路径兜底，避免同一件事计两次
    st["will_llm"] = (st.get("llm_chat") is not None
                      and not (st["parsed"] is not None
                               and looks_like_question(st["text"])))
    return {}


def node_perceive(dh, st):
    """听见 + 立刻落盘。**这是中断安全的地基**：此后无论发生什么，
    「用户说过这句话」都不会丢。"""
    ho = dh.hear(st["text"], speaker="用户", t=st["t"],
                 valence=0.0 if st["will_llm"] else None)
    _ensure_heard(dh.store, ho, st["text"], "用户", st["t"])
    ho["trust"] = _trust(st["text"], st["parsed"], dh)
    dh.store.commit()                      # ← 用户的话在此刻落盘（~200 µs）
    st["hear_out"] = ho
    return {}


def node_recall(dh, st):
    """三路检索一次，结果进 st['ctx']。下游全部消费它，不再自己查。"""
    st["ctx"] = collect_context(dh, st["text"], parsed=st["parsed"],
                                t_now=dh.t)
    return {}


def node_plan(dh, st):
    """决定走哪条腿（policy.decide：决策读记忆状态），并把 prompt 组装好。

    prompt 组装的范围：ack/defer/offline 三条腿不需要（离线没有 RAG 可拼）；
    fast 腿**也**预组装——点查命中时那 ~300 µs 排版是已知的浪费价，
    换的是「弃权降级」不用在锁外补一次锁内排版（compose 出锁了，
    临时拿锁排版会拉长临界区）。两害相权取其轻。

    **情绪在这里第一次改控制流**（不只是措辞）：policy 判 defer
    （说不动）且这句话值是应答词时，不再花 3-10 秒调 LLM 硬凑一段话——
    按人的样子应一声就完了。这不是偷懒优化：实测「发困/低落」状态下
    强行让模型展开，产出的是最长、最不像的那种。
    """
    st["route"], st["route_why"] = policy.decide(dh, st)   # try_fast/disp 已落回 st
    if st["route"] in ("ack", "defer", "offline"):
        return {}                          # 离线没有 RAG 可拼；ack/defer 不需要
    # 走 LLM 的话（含 fast 弃权降级）先把 prompt 备好（组装要读 ctx，所以在锁内）
    if st.get("llm_chat") is not None:
        prompt = llm_bridge.build_system_prompt(dh, st["text"], ctx=st["ctx"],
                                                parsed=st["parsed"])
        if st.get("vision_desc"):
            prompt += "\n\n【你眼睛此刻看到的画面】" + st["vision_desc"]
        st["prompt"] = prompt
    return {}


def node_compose(dh, st):
    """不持锁的生成节点。五条腿：ack / defer / fast(点查) / llm(RAG) / offline。

    `defer` 分支是**情绪决定不做**：低唤醒 + 没耐心 + 这句话既解析不出结构
    也没检索到任何东西时，不调 LLM 硬凑。人发困或烦的时候就是这么应的。
    """
    route = st.get("route", "offline")
    if route == "ack":
        st["reply"], st["flags"], st["sources"] = st["ack"], [], []
        st["used_memory"], st["stale"], st["cited"] = True, False, []
        st["leg"] = "ack"
        return {}
    if route == "defer":
        st["reply"] = policy.defer_line(dh, st.get("disp"))
        st["flags"], st["sources"], st["cited"] = [], [], []
        st["used_memory"], st["stale"] = False, False
        st["deferred"] = True
        st["leg"] = "defer"
        return {}
    # fast 与 offline 共用点查腿：route=="fast" 先点查；offline 且句形像
    # 事实问句（try_fast）也先点查——离线模式下记忆问答照常工作
    if st.get("try_fast"):
        ans, phr, diag = dh.ask(st["text"], record=False)
        if not ans.abstain:
            srcs = [s for s in (ans.sources or []) if s not in ("用户", "我")]
            st["reply"] = phr
            st["flags"] = diag["conflict_flags"]
            st["sources"] = srcs
            st["stale"] = bool(ans.stale)
            st["cited"] = list(ans.fact_ids or [])
            st["used_memory"] = True
            st["leg"] = "fast_hit"
            return {}
        # 点不中就掉进 LLM——这是原来写成 elif 会漏掉的路径
        if st.get("llm_chat") is None:
            st["reply"], meta = llm_bridge.offline_reply(dh, st["text"],
                                                         record=False)
            st["flags"], st["sources"], st["cited"] = [], [], []
            st["used_memory"], st["stale"] = meta["used_memory"], False
            st["leg"] = "fast_fallback_offline"
            return {}
    # L2 问句解析兜底：L0/L1 没路由出来 + 配了 Key + 问句形 + 句里有可查
    # 主体（「我」或实体）→ 问一次老师。解析合法即回填线索（自学习，finish
    # 落库），点查命中就直接答——省掉整趟 RAG，措辞走来源分级。
    # 开放问句（无主体，如「你觉得生活有什么意义」）不问老师——解析必然
    # 失败，白烧一次调用（回归实测抓过）。
    if (st.get("llm_chat") is not None and st.get("parsed") is None
            and looks_like_question(st["text"])
            and ("我" in st["text"] or dh.ex.find_entities(st["text"]))):
        parsed2, ans2 = llm_bridge.llm_route_question(dh, st["llm_chat"],
                                                      st["text"])
        if parsed2 is not None:
            st["learn_hint"] = (parsed2["predicate"], st["text"])
            if not ans2.abstain:
                disp = st.get("disp") or _disp(dh)
                srcs = [s for s in (ans2.sources or []) if s not in ("用户", "我")]
                st["reply"] = ans2.phrase(directness=disp.directness)
                st["flags"], st["sources"] = [], srcs
                st["stale"] = bool(ans2.stale)
                st["cited"] = list(ans2.fact_ids or [])
                st["used_memory"] = True
                st["leg"] = "llm_parse_hit"
                return {}
        # 没答上/没解析出：线索照学（finish 落库），落回完整 LLM
    if st.get("llm_chat") is not None:
        try:
            raw = (st["llm_chat"](st["prompt"], st["text"]) or "").strip()
        except Exception as exc:       # noqa: BLE001  网络抖动不能杀生成线程：
            st["llm_error"] = str(exc)[:200]   # 回复留空，finish 的离线兜底接住
            raw = ""
        st["raw"] = raw
        st["reply"], st["apr"] = llm_bridge.parse_appraisal(raw)
        st["flags"], st["sources"], st["cited"] = [], [], []
        st["used_memory"], st["stale"] = True, False
        st["leg"] = ("llm_error" if st.get("llm_error")
                     else "fast_fallback_llm" if route == "fast" else "llm")
        return {}
    st["reply"], meta = llm_bridge.offline_reply(dh, st["text"], record=False)
    st["flags"], st["sources"], st["cited"] = [], [], []
    st["used_memory"], st["stale"] = meta["used_memory"], False
    st["leg"] = "offline"
    return {}


def node_affect(dh, st):
    """LLM 评价落情绪。评价是「对小星自己的冲击」，所以必须在
    自己的话入记忆之前——否则 TTS 用的还是上一轮的情绪。"""
    if st.get("route") == "ack":
        return {}
    llm_bridge.apply_appraisal(dh, st.get("apr"), st["text"])
    return {}


def node_finish(dh, st):
    """自己的话也记 + 边界提交。空回复用离线兜底接住（LLM 偶发空串）。"""
    if not st["reply"]:
        st["reply"], meta = llm_bridge.offline_reply(dh, st["text"],
                                                     record=False)
        st["used_memory"] = meta["used_memory"]
    t2 = time.time() + 0.01
    ho2 = dh.hear(st["reply"], speaker="小星", t=t2)
    _ensure_heard(dh.store, ho2, st["reply"], "小星", t2)
    st["hear_out"]["obs_ids"] = ((st["hear_out"].get("obs_ids") or [])
                                 + (ho2.get("obs_ids") or []))
    # L2 学到的问法线索在这里落库（同一次 commit；学不成不挡回话）
    hint = st.pop("learn_hint", None)
    if hint:
        try:
            dh.qrouter.learn(hint[0], hint[1], t=t2)
        except Exception:                  # noqa: BLE001
            pass
    dh.store.commit()                      # 小星的话落盘
    return {}


def node_speak(dh, st):
    """TTS 派发交给调用方（CLI 是丢线程，web 是攒批）。这里只准备参数。"""
    from voice import tone as tone_mod
    sp = dh.affect.outlets()["speech"]
    st["prosody"] = dict(sp, tone=tone_mod.pick_tone(dh.affect.state))
    st["mood"] = _mood(dh)
    return {}


_NODE_FN = dict(guard=node_guard, perceive=node_perceive, recall=node_recall,
                plan=node_plan, compose=node_compose, affect=node_affect,
                finish=node_finish, speak=node_speak)
