# -*- coding: utf-8 -*-
"""可选的 LLM 桥：把记忆系统的产出接到 main.py 的 think 步。

纪律 [GRAPH §5]：LLM 只在最后一步出现——前面三步（点查/语义/扩散）
都是数学。本模块做三件事：
    1. 把检索结果按来源分级写成系统提示词里的一段「记忆上下文」；
    2. 把情绪四出口写成一行语气指令（韵律给 TTS，文字只读 P/D）；
    3. respond()：一轮完整对话（听见->点查/LLM->写回），main.py 的
       think 节点只调这一个函数；LateExtractor：正则没吃下的事实
       后台补记（与 web/server.py 同一套五道闸门）。

没有网络 / 没有 API key 时整个系统照常工作（demo 与测试都不经过这里）。
"""
import threading
import time

from . import config
from .tick import looks_like_question


def memory_context_block(dh, question=None, k=8):
    """给 LLM 的记忆上下文：当前相关事实（带来源分级与有效期措辞）。"""
    lines = []
    # 相关实体 = 问题里抽到的实体（先走完整解析，解析不动就按名字找）；
    # 都没有才兜底到最近写过事实的那个
    ents = []
    if question:
        parsed = dh.ex.parse_question(question) or dh.ex.parse(question)
        if parsed is not None:
            ents = [parsed["subject"]]
        else:
            found = dh.ex.find_entities(question)
            if found:
                ents = [found[0][2]]              # 问题点名的实体，不必成句
    if not ents:
        row = dh.store.con.execute(
            "SELECT entity FROM fact WHERE status='active' "
            "ORDER BY written_t DESC LIMIT 1").fetchone()
        if row is None:
            return "（记忆里还没有任何事实。）"
        ents = [row["entity"]]
    for ent in ents:
        rows = dh.store.con.execute(
            "SELECT predicate,value,status,rel,valid_to FROM fact "
            "WHERE entity=? AND status != 'retracted' "
            "ORDER BY written_t DESC LIMIT ?", (ent, k)).fetchall()
        for r in rows:
            rel = int(r["rel"])
            tag = config.REL_LABEL.get(rel, "?")
            if r["status"] != "active":
                lines.append("【%s·已失效】%s的%s曾是%s" % (tag, ent, r["predicate"], r["value"]))
            else:
                lines.append("【%s】%s的%s是%s" % (tag, ent, r["predicate"], r["value"]))
    return "\n".join(lines)


def tone_block(dh):
    """情绪四出口 -> 一行语气指令（文字只读 P/D，韵律给 TTS）。"""
    st = dh.affect.state
    lang = dh.affect.outlets()["language"]
    speech = dh.affect.outlets()["speech"]
    mood = "平静"
    if st.p < -0.3:
        mood = "有点低落" if st.a < 0.4 else "明显不高兴"
    elif st.p > 0.3:
        mood = "心情不错" if st.a < 0.4 else "很兴奋"
    return ("【当前状态】%s（语气偏%s、说话强势度%.1f）；"
            "TTS 韵律：语速%.2f 音高%.2f 音量%.2f"
            % (mood, "强势" if lang["assertiveness"] > 0 else "收敛",
               lang["assertiveness"], speech["rate"], speech["pitch"],
               speech["energy"]))


def build_system_prompt(dh, question):
    lines = [
        "你是一个叫小星的虚拟人。说话自然、口语化，每次回复不超过两句话。",
        "记忆里的信息按来源分级措辞：【亲眼看到】可以直接说，"
        "【亲口告诉】要说「你告诉过我」，【听说】要说「听谁说」，"
        "【推断】必须说「我猜」，【先验】不能当记忆说。",
        "已失效的信息绝不能当现在的事实端出来，要说「我记得以前是…现在不确定」。",
    ]
    owner = dh.store.get_owner()
    if owner:
        # 对话对象身份：没有这行，LLM 会把记忆里别人（老王）的事实安到用户头上
        lines.append("正在和你对话的人是【%s】：记忆里关于%s的事就是对方本人的事，"
                     "对方说「我」就是%s；其他实体是别人，"
                     "不要把别人的事实说成对方的。" % (owner, owner, owner))
    lines += [
        "",
        memory_context_block(dh, question),
        "",
        tone_block(dh),
    ]
    return "\n".join(lines)


def offline_reply(dh, text):
    """没配 Key 也能聊：事实问答走记忆点查（措辞现成），其他给克制的回应。"""
    parsed = dh.ex.parse_question(text)
    if parsed is not None and looks_like_question(text):
        ans, phr, _ = dh.ask(text)
        return phr, dict(used_memory=True)
    parsed2 = dh.ex.parse(text)
    if parsed2 is not None:
        return "嗯，我记下了：%s的%s是%s。" % (parsed2["subject"],
                                             parsed2["predicate"],
                                             parsed2["value"]), dict(used_memory=True)
    if any(w in text for w in ("你好", "在吗", "嗨", "hello", "hi")):
        return "在呢。配好 API Key 之后我就能聊得自然些，现在只带着记忆跟你说话。", \
            dict(used_memory=False)
    return "（离线模式）这句话我记住了。想让我正常聊天，点右上角配置 API Key。", \
        dict(used_memory=False)


def respond(dh, text, llm_chat=None, vision_desc=None):
    """一轮完整对话：听见 -> 快路径点查 / 慢路径 LLM -> 自己的话也记 -> 落盘。

    llm_chat(system, user)->str 为 None 时走离线兜底；vision_desc 是
    眼睛看到的画面（有它就不走记忆快路径——问「这是什么」要看眼睛）。

    返回 dict(reply, mood, prosody, flags, sources, used_memory, stale,
              hear_out)。prosody 是情绪四出口的语音三项，喂给 TTS。
    """
    with dh.store.lock:
        t = time.time()
        # 1 它先听见：感知/注意/评估/写回（情绪与记忆在此更新）
        hear_out = dh.hear(text, speaker="用户", t=t)
        # 2 快路径：事实类问句点查记忆，不花 token（来源措辞现成）
        parsed = dh.ex.parse_question(text)
        fast = None
        if (parsed is not None and looks_like_question(text)
                and vision_desc is None):
            ans, phr, diag = dh.ask(text)
            if not ans.abstain:
                srcs = [s for s in (ans.sources or []) if s not in ("用户", "我")]
                fast = dict(reply=phr, flags=diag["conflict_flags"],
                            sources=srcs, stale=bool(ans.stale),
                            degraded=bool(ans.degraded),
                            cited=list(ans.fact_ids))
        # 3 慢路径：LLM 组织语言（记忆上下文 + 情绪语气注入）
        if fast is not None:
            reply, flags, srcs = fast["reply"], fast["flags"], fast["sources"]
            used_memory, stale = True, fast["stale"]
        elif llm_chat is not None:
            prompt = build_system_prompt(dh, text)
            if vision_desc:
                prompt += "\n\n【你眼睛此刻看到的画面】" + vision_desc
            reply = (llm_chat(prompt, text) or "").strip()
            flags, srcs, used_memory, stale = [], [], True, False
        else:
            reply, meta = offline_reply(dh, text)
            flags, srcs = [], []
            used_memory, stale = meta["used_memory"], False
        if not reply:                       # LLM 偶发空串 -> 离线兜底接住
            reply, meta = offline_reply(dh, text)
            used_memory = meta["used_memory"]
        # 4 自己说的话也记（self 的陈述 rel=1）
        hear_out_reply = dh.hear(reply, speaker="小星", t=time.time() + 0.01)
        hear_out["obs_ids"] = ((hear_out.get("obs_ids") or [])
                               + (hear_out_reply.get("obs_ids") or []))
        dh.store.commit()                   # 消息即落盘（commit 后 PASSIVE checkpoint）
        st = dh.affect.state
        return dict(reply=reply,
                    mood=dict(p=round(st.p, 3), a=round(st.a, 3), d=round(st.d, 3)),
                    prosody=dict(dh.affect.outlets()["speech"]),
                    flags=flags, sources=srcs, used_memory=used_memory,
                    stale=stale, hear_out=hear_out,
                    cited=list(fast["cited"]) if fast else [])


class LateExtractor:
    """LLM 补记：正则快路没吃下、但句子可能有信息 -> 后台五道闸门抽取写回。

    不阻塞回复；同一时刻只跑一条（上一条没完就丢给下一条的概率优先）。
    web/server.py 的 _late_extract 与 main.py 共用这一套。
    """

    def __init__(self, dh, chat_fn, on_written=None):
        self.dh = dh
        self.on_written = on_written        # fact_ids 回调（web 用来点亮识海）
        from .extract_llm import LLMExtractor, default_registry
        self.ex = LLMExtractor(chat_fn, registry=default_registry(),
                               extractor=dh.writer.ex)
        self._busy = threading.Lock()

    def submit(self, text, t, speaker, hear_out):
        """ hear_out 里正则已经写出了事实就不再补记。返回线程（测试 join 用）。"""
        if hear_out.get("written"):
            return None
        th = threading.Thread(target=self._run,
                              args=(text, t, speaker, hear_out),
                              daemon=True)
        th.start()
        return th

    def _run(self, text, t, speaker, hear_out):
        if not self._busy.acquire(blocking=False):
            return                           # 上一条还没抽完，丢给下一条
        try:
            topics = [it["text"] for it in self.dh.wm.items[-2:]]
            cands, _ = self.ex.extract(text, topics)
            if not cands:
                return
            with self.dh.store.lock:
                obs_ids = hear_out.get("obs_ids") or []
                obs_id = obs_ids[0] if obs_ids else None
                if obs_id is None:           # 问句路径没有 writer hear——补一条 obs
                    obs_id = self.dh.store.add_obs(t, "told", "text", text,
                                                   source=speaker, rel=1, seg_bnd=0.0)
                def _learn(name):
                    if self.dh.writer.ex.learn_alias(name):
                        self.dh.store.alias_learn(name, name)
                out = self.dh.writer.hear_llm_candidates(
                    text, t, speaker, obs_id, cands, learn_entity_cb=_learn)
                if out["fact_ids"]:
                    self.dh.store.commit()   # 后台写入自己落盘
                    if self.on_written is not None:
                        self.on_written(out["fact_ids"])
        except Exception:                    # noqa: BLE001
            pass                             # obs 里原文永远在，丢了可重来
        finally:
            self._busy.release()
