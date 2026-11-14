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
import json
import re
import threading
import time

from . import config
from .limbic.emotion import Appraisal, appraise_to_pad
from .limbic.disposition import gate as emo_gate
from .extract import self_intro_name
from .core.tick import looks_like_question


def seen_fact_ids(dh, question, k=8):
    """问题相关的事实行 id（精确路 + 语义路），给图扩展当种子。"""
    ids = []
    if not question:
        return ids
    parsed = dh.qrouter.route(question) or dh.ex.parse(question)
    ents = [parsed["subject"]] if parsed is not None else []
    if not ents:
        found = dh.ex.find_entities(question)
        if found:
            ents = [found[0][2]]
    for ent in ents:
        rows = dh.store.con.execute(
            "SELECT fact_id FROM fact WHERE entity=? AND status='active' "
            "ORDER BY written_t DESC LIMIT ?", (ent, k)).fetchall()
        ids += [int(r["fact_id"]) for r in rows]
    try:
        _g = getattr(dh, "growth", None)
        _n = _g.traits.get()["N"] if _g is not None else None
        for kind, rid, _s in dh.retriever.recall(query_text=question, mood=0.0,
                                                 t_now=dh.t, k_gate=k,
                                                 n_trait=_n):
            if kind == "fact":
                ids.append(int(rid))
    except Exception:
        pass
    return list(dict.fromkeys(ids))


def memory_context_block(dh, question=None, k=8, ctx=None):
    """给 LLM 的记忆上下文。ctx 由 dh/context.py 一次收集，这里只排版。

    为什么拆出去：原先这条链上跑三次 recall（seen_fact_ids 一次、
    这里一次、dh.ask 内部一次），同一件事查三遍，共 ~660 µs。根因不是
    慢，是**结果没有地方存**——谁要谁自己再算一遍。

    ctx 为 None 时退回老的自己查（向后兼容 web/chat_service.py 与测试）；
    走 turn.run() 的路径一律传 ctx，一次对话只召回一次。
    """
    from .context import collect_context, render_memory_block
    if ctx is None:
        ctx = collect_context(dh, question, k=k)
    return render_memory_block(dh, ctx)


def tone_block(dh):
    """情绪四出口 -> 语气指令（文字读 P/D + 关系信任，韵律给 TTS）。

    这一段是**行为约束**，不只是形容词：`disposition.prompt_line()` 里的
    每一条都是祈使句（「别超过十个字」「先说『我记得』」），直接进 LLM 的
    指令区。之前只有「有点低落/心情不错」这类描述，模型可以读完照旧说三
    段话——描述不构成约束。
    """
    from .limbic.persona import Persona
    from .limbic.disposition import disposition
    from voice import tone as tone_mod
    st = dh.affect.state
    lang = dh.affect.outlets()["language"]
    name = tone_mod.pick_tone(st)
    disp = disposition(dh)
    tr = Persona(dh).trust()
    stance = ("对用户有点戒备、话留三分" if tr < 0.4 else
              "和用户很亲近、可以开玩笑" if tr > 0.75 else
              "把用户当朋友")
    head = ("【当前状态】%s（语气偏%s、说话强势度%.1f）；%s；"
            "现在说话的语气是「%s」；%s"
            % (disp.label, "强势" if lang["assertiveness"] > 0 else "收敛",
               lang["assertiveness"], stance,
               tone_mod.TONE_LABEL[name],
               tone_mod.describe_for_prompt(st)))
    behavior = disp.prompt_line()
    return head + ("\n" + behavior if behavior else "")


def recent_dialogue_block(dh, question=None, k=10, width=60):
    """短期上下文：最近 k 条对话（经历层 told+text 的用户/小星发言，滚动）。

    长期靠双时态记忆（不是窗口），短期靠这里——回答「上下文怎么做的」：
    不是一个大窗口聊到死，是「事实层 + 滚动近况」两层拼出来的。"""
    rows = dh.store.con.execute(
        "SELECT source, payload FROM obs "
        "WHERE kind='told' AND channel='text' AND source IN ('用户','小星') "
        "ORDER BY obs_id DESC LIMIT ?", (k + 1,)).fetchall()
    rows = [r for r in reversed(rows)]               # 时间正序
    lines = []
    for r in rows:
        if question and r["payload"] == question:
            continue                                  # 当前这句由 user 消息承载
        who = "用户" if r["source"] == "用户" else "小星"
        lines.append("%s：%s" % (who, r["payload"][:width]))
    if not lines:
        return ""
    return "\n".join(["【最近对话】（短期上下文，滚动窗口）"] + lines[-k:])


def register_owner_if_intro(dh, text):
    """「我叫X」/「我是X」自报家门 -> 登记属主（「我」的确定性消解锚）。

    没有属主时，writer 对「我」的陈述只能整句拒写（宁可漏记）——用户说
    「老王是我的朋友」连不上线，根因就是这里空着。返回确认语（无 intro
    或无变化返回 None）；改名时明说，登记错误用户再说一句就覆盖。"""
    name = self_intro_name(text)
    if not name:
        return None
    prev = dh.store.get_owner()
    dh.store.set_owner(name, note="self-intro")
    if dh.writer.ex.learn_alias(name, name):
        dh.store.alias_learn(name, name)              # 新实体进别名表，正则下次直接命中
    if prev is None:
        return "记住了，你是%s。以后你说「我」，我都知道是谁。" % name
    if prev != name:
        return "好，从现在起你是%s（原来记的是%s）。" % (name, prev)
    return None


def build_system_prompt(dh, question, ctx=None, parsed=None):
    """系统提示词。ctx 由 dh/context.py 的 collect_context 产出。

    ctx 为 None 时退回老路（自己 collect 一次）——保持向后兼容：
    网页端和测试还在用 `build_system_prompt(dh, text)`。
    但走 turn.run() 的路径一律传 ctx，一次对话只召回一次。
    """
    if ctx is None:
        from .context import collect_context
        ctx = collect_context(dh, question, parsed=parsed)
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
        memory_context_block(dh, question, ctx=ctx),
    ]
    dialogue = recent_dialogue_block(dh, question)
    if dialogue:
        lines += [dialogue]
    lines += ["", _appraisal_instruction(dh), "", tone_block(dh)]
    return "\n".join(lines)


def _appraisal_instruction(dh):
    """情绪评价协议：LLM 按「活性格」给评价维度打分，数学换算成情绪。

    性格不是出厂定死的——这里注入的是 Persona.summary_text()：
    出厂种子 + 被经历塑形后的基线/信任/自我认知。不让 LLM 直接输出
    情绪标签——它指认（好不好/谁的锅/多意外/多相关），appraise_to_pad
    + OU 负责结论。评分挂在回复末尾的标签里，零额外调用。"""
    from .limbic.persona import Persona
    return (
        "【你的性格（出厂种子 + 经历塑形，会随经历缓慢改变）】\n"
        + Persona(dh).summary_text() + "\n"
        "每次回复的最末尾，另起一行输出你对用户这句话的情绪评价，"
        "严格按这个格式、不要输出别的解释：\n"
        '<appraisal>{"desirability":0.0,"agency":"circumstance",'
        '"novelty":0.2,"goal_relevance":0.8}</appraisal>\n'
        "字段含义：desirability -1~1（这句话对你是好事还是坏事）；"
        'agency 取 "self"/"other"/"circumstance"（事情是谁导致的：被骂=other，'
        "自己搞砸=self，天意=circumstance）；novelty 0~1（多意外）；"
        "goal_relevance 0~1（跟你的自尊/目标的关联有多强）。"
        "按你当前的性格状态打分：被夸明显开心、被骂会委屈或生气、平淡问候接近 0。")


_APPRAISAL_RE = re.compile(r"<appraisal>\s*(\{.*?\})\s*</appraisal>", re.S)
_APPRAISAL_LOOSE_RE = re.compile(r"\{[^{}]*desirability[^{}]*\}", re.S)


def parse_appraisal(reply):
    """从回复里剥离 <appraisal> 标签并解析成 Appraisal，字段严格钳住。

    标签缺失/坏 JSON -> (原文, None)——调用方退回词典兜底。
    返回 (干净回复文本, Appraisal|None)。"""
    m = _APPRAISAL_RE.search(reply)
    if m:
        blob, clean = m.group(1), (reply[:m.start()] + reply[m.end():]).strip()
    else:
        m2 = _APPRAISAL_LOOSE_RE.search(reply)
        if not m2:
            return reply, None
        blob, clean = m2.group(0), reply.replace(m2.group(0), "").strip()
    try:
        data = json.loads(blob)

        def _f(key, lo, hi, default):
            try:
                return max(lo, min(hi, float(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        agency = str(data.get("agency", "circumstance"))
        if agency not in ("self", "other", "circumstance"):
            agency = "circumstance"
        apr = Appraisal(desirability=_f("desirability", -1, 1, 0.0),
                        likelihood=1.0, agency=agency,
                        novelty=_f("novelty", 0, 1, 0.2),
                        goal_relevance=_f("goal_relevance", 0, 1, 0.8))
        return clean, apr
    except (ValueError, TypeError):
        return reply, None


def apply_appraisal(dh, apr, user_text, already_sourced=False):
    """把评价落到情绪系统（drive 注入，锁外算、锁内应用由调用方保证）。

    LLM 没给评价时退回小词典——离线模式全走词典；在线模式词典只是兜底。

    一句话至多一个效价来源（审查 P1-2）：调用方从本句 hear 的返回值当场
    捕获 valence_sourced（词典效价≠0，离线/已解析问句路径）并经
    already_sourced 传入时，这里**整体跳过**——词典兜底和 LLM 标签都不
    再叠加，成长证据只留 hear 落下的那一条、情绪只吃 hear 那一次驱动。
    每句一份、随 hear 返回值走：不挂 dh 共享字段，插话（LLM 飞行窗口内
    第二句 hear）覆盖不了上一句的判定。web 在线主路径先听成中性、
    sourced=False，评价照旧由这里单独承担。

    D 维（支配感）以前塌成一个符号函数：`agency="other" if valence<0 else
    "circumstance"`，controllability 从不赋值（默认 0.5），实测 D 只有两个
    取值——负面 -0.280、正面 0.000。于是 outlet_language 的
    `assertiveness = pad.d` 实际上只反映「刚才是不是坏事」。这里按
    「谁的责任」区分：自责（self，D 升）与被压制（other，D 降）都是坏事，
    但一个让人更笃定（我搞砸了，清楚），一个让人更虚（他骂我，没底）。
    """
    if already_sourced:
        return None
    if apr is None:
        v = dh.valence_fn(user_text)
        if v == 0:
            return None
        # 词典极性只有正负，分不清责任归属：负面一律按「外部压制」处理
        # （被骂、被拒），不假设自己搞砸——过度自责比过度防御更不像人。
        apr = Appraisal(desirability=v, likelihood=1.0,
                        agency="other" if v < 0 else "circumstance",
                        novelty=0.2, goal_relevance=0.8,
                        controllability=0.35 if v < 0 else 0.6,
                        certainty=0.6)
    delta = appraise_to_pad(apr)
    if abs(delta.p) + abs(delta.a) + abs(delta.d) < 1e-6:
        return None
    with dh.store.lock:
        # 情绪冲击一次到位（不再乘 0.15 阻尼）：被狠骂一记当场变脸，
        # 恢复交给之后的 decay——人的情绪事件曲线就是 shock + decay。
        # 旧实现走 drive 路径被 (1-a)≈0.15 阻尼，骂一记只落 8%，
        # 出厂亮底色下「骂了不生气」（用户实测抓出）
        dh.affect.shock(delta)
        # 评价事件本身也入经历层（valence=desirability）——性格塑形、
        # 时光回溯都从这里取证据；obs 追加，可溯源。
        # 编码强度在评价落定后补算：此刻的 PAD 才是这一轮情绪的真实读数
        # （web/CLI 慢路径下用户原话按纪律先听成中性，情绪住在这行里）。
        # 存侧性格加权与 tick 第 6 步同一根线：尽责（C）编码更牢
        _g = getattr(dh, "growth", None)
        _boost = _g.traits.encode_factor() if _g is not None else 1.0
        sal = min(1.0, max(0.0, emo_gate(dh.affect.state,
                                         surprise=float(apr.novelty),
                                         attention=1.0) * _boost))
        dh.store.add_obs(time.time(), "infer", "appraisal",
                         json.dumps({"desirability": round(apr.desirability, 2),
                                     "agency": apr.agency,
                                     "novelty": round(apr.novelty, 2),
                                     "goal_relevance": round(apr.goal_relevance, 2)},
                                    ensure_ascii=False),
                         source="小星", rel=3, seg_bnd=0.0,
                         valence=apr.desirability, salience=sal)
    return delta


def llm_route_question(dh, chat_fn, text):
    """L2 问句解析兜底：LLM 只当老师，解析合法就把问法回填线索库。

    调用时机：L0/L1 路由不出来 + 配了 Key + 问句形。纪律与抽取五道闸门
    同款——predicate 必须在注册表（防幻觉谓词），subject 必须是属主/别名
    实体/原文实体（防张冠李戴）；解析不合法 → (None, None)，什么都不学。

    返回 (parsed|None, ans)：parsed 非 None 时问法本身值得学（由调用方
    回填，答没答得上另说）；ans 非 abstain 时直接点查命中，省掉整趟 RAG。
    """
    preds = [r["predicate"] for r in
             dh.store.con.execute("SELECT predicate FROM pred")]
    owner = dh.store.get_owner()
    system = (
        "你是问句解析器。把问句解析成「查谁、查什么」。规则：\n"
        "1. predicate 只能选：" + "、".join(sorted(preds)) + "\n"
        "2. subject：句中的「我」指" + (owner or "说话人本人（填「我」）") +
        "；其余用原文里出现的实体名；解析不出主体就填空串。\n"
        "3. 不是在查某实体的某项信息（比如闲聊、寒暄）→ 两个字段都填空串。\n"
        '只输出 JSON：{"subject":"","predicate":""}'
    )
    raw = (chat_fn(system, "问句：" + text) or "").strip()
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    except (ValueError, json.JSONDecodeError):
        return None, None
    subject = str(data.get("subject", "")).strip()
    predicate = str(data.get("predicate", "")).strip()
    if not predicate or predicate not in preds or not subject:
        return None, None
    if subject == "我" and owner:
        subject = owner              # 模型偷懒回「我」也照消解（与 L0 同一规则）
    canon = dh.ex.canon(subject)
    if canon:
        subject = canon
    elif subject != (owner or "") and subject not in text:
        return None, None              # 主体不落地：宁可漏，不猜
    ans = dh.retriever.ask_fact(subject, predicate)
    parsed = dict(src=None, subject=subject, predicate=predicate, value=None,
                  entities=dh.ex.find_entities(text), change=False,
                  body=text, via="llm")
    return parsed, ans


def offline_reply(dh, text, record=True):
    """没配 Key 也能聊：事实问答走记忆点查（措辞现成），其他给克制的回应。

    record=False：这句问话的原话已经由调用方写进经历层（web 的
    _ensure_heard），别再让 ask 写第二遍。"""
    parsed = dh.qrouter.route(text)
    if parsed is not None and looks_like_question(text):
        ans, phr, _ = dh.ask(text, record=record)
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
    """一轮完整对话。薄封装：真正的节点实现搬到 dh/brain/（主图）。

    保留这个入口是因为它是「一轮对话」的稳定契约，main.py / 测试 / 将来的
    HTTP 适配器都只认它。旧的十阶段实现有三个病（都实测过）：
        1  一把锁罩住全程，LLM 那 3-10 秒也在锁里；
        2  commit 在最末尾——用户的话要等 LLM 回来才落盘，打断一次丢一句；
        3  中间状态全是局部变量，外部看不到也测不了。
    brain.run() 逐条修掉，同时保持返回字段一模一样。
    """
    from .brain import run
    return run(dh, text, llm_chat=llm_chat, vision_desc=vision_desc)

class LateExtractor:
    """LLM 补记：正则快路没吃下、但句子可能有信息 -> 后台五道闸门抽取写回。

    不阻塞回复；同一时刻只跑一条（上一条没完就丢给下一条的概率优先）。
    web/server.py 的 _late_extract 与 main.py 共用这一套。
    """

    def __init__(self, dh, chat_fn, on_written=None):
        self.dh = dh
        self.on_written = on_written        # fact_ids 回调（web 用来点亮识海）
        from .extract.llm import LLMExtractor, default_registry
        self.ex = LLMExtractor(chat_fn, registry=default_registry(),
                               extractor=dh.writer.ex)
        self._busy = threading.Lock()

    def submit(self, text, t, speaker, hear_out):
        """判据：正则这次抽得**可信**才跳过补记。返回线程（测试 join 用）。

        原来的判据是 `if hear_out.get("written")`——只要正则写出一条事实就
        不再补记。问题是**抽错的时候 written 也非空**：实测
        `张姨说老王出院了` 写出 `张姨|病情|出院`（主语记错人）、
        `老王的电话是协和医院心内科` 写出 `老王|工作单位|协和医院`
        （值截断），两条都 written 非空、都错，而 LLM 永不启动纠错——
        Tier 0 把 Tier 1 关掉了。

        改成看 dh/brain/nodes.py 的 `_trust()`：trusted 才跳过；suspect（主语
        记错 / 值被截断）和 empty（一条都没写）都交给 LLM。
        """
        if hear_out.get("trust") == "trusted":
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
