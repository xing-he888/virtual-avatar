# -*- coding: utf-8 -*-
"""一轮对话的**检索上下文**：一次收集，多处消费。

为什么单独一层：原先 `build_system_prompt` 一条链上跑了三次 recall
（`seen_fact_ids` 一次、`memory_context_block` 一次、`dh.ask` 内部一次），
同一件事查了三遍，共 ~660 µs。根因是**没有地方存上一次的结果**——
结果活在各个函数的局部变量里，谁要谁自己再算一遍。

这个模块把「查」和「排版」分开：

    collect_context(dh, question)   → ctx        查一次，产出全部原料
    render_memory_block(dh, ctx)    → str        纯格式化，零查询

ctx 是普通 dict，可以塞进 TurnState 跨节点传，也可以被测试直接构造。

三路召回的分工（缺一路就漏，这是实测结论，不是设计美学）：
    A 精确路  问题点名实体的**全部**事实（含已失效行）——「问什么答什么」，
              也是唯一能把 superseded 行带出来的路，措辞降级全靠它。
    B 语义路  向量 top-k，命中 fact（结构化）与 obs（原话）两种 kind。
              obs 不能丢：正则抽不出的句子在 fact 层根本不存在。
    C 图路    从种子事实扩散 2 跳找**别的实体**——「顺着话头聊下去」的原料。
"""
from . import config


def _resolve_entities(dh, question, parsed=None):
    """问题里点到的实体。问句认不出就是空——不拿别人的档案凑数。

    历史上这里有个致命的兜底：认不出实体时退回「最近写的一条事实的实体」。
    实测「我姐在哪」（别名词典无「我姐」，parse_question 返回 None）因此
    端出了张姨的住址和电话。错答比不答更糟，所以只有**没有问句**
    （自由陈述场景）才允许用最近实体兜底。
    路由器版 parse_question（属主消解 + 自学问法）从这里过——「我多少岁」
    的 A 路从此能精确点名属主。
    """
    if not question:
        return []
    if parsed is None:
        parsed = dh.qrouter.route(question) or dh.ex.parse(question)
    if parsed is not None:
        return [parsed["subject"]]
    found = dh.ex.find_entities(question)
    return [found[0][2]] if found else []


def collect_context(dh, question=None, k=None, parsed=None, t_now=None):
    """一次把 A/B/C 三路的原料全取出来。

    返回 dict：
        ents     [实体]                  A 路点到的实体
        facts    [(ent,pred,val,st,rel)] 结构化事实（已去重，顺序即排版顺序）
        spoken   [(who, said)]           原话（已去重）
        related  [(fact_id, act)]        C 路联想
        hits     [(kind, rid, score)]    语义路原始命中
        seeds    [fact_id]               C 路的种子
    """
    k = config.GATE_K if k is None else k
    t_now = dh.t if t_now is None else t_now
    ctx = dict(ents=[], facts=[], spoken=[], related=[], hits=[], seeds=[])

    # ---- A 精确路
    ents = _resolve_entities(dh, question, parsed)
    if not ents and not question:
        row = dh.store.con.execute(
            "SELECT entity FROM fact WHERE status='active' "
            "ORDER BY written_t DESC LIMIT 1").fetchone()
        if row is not None:
            ents = [row["entity"]]
    ctx["ents"] = ents

    seen = set()

    def _add_fact(ent, pred, val, status, rel):
        key = (ent, pred, val)
        if key in seen:
            return
        seen.add(key)
        ctx["facts"].append((ent, pred, val, status, int(rel or 0)))

    for ent in ents:
        rows = dh.store.con.execute(
            "SELECT predicate,value,status,rel FROM fact "
            "WHERE entity=? AND status != 'retracted' "
            "ORDER BY written_t DESC LIMIT ?", (ent, k)).fetchall()
        for r in rows:
            _add_fact(ent, r["predicate"], r["value"], r["status"], r["rel"])
        # 本人改口协议（回答层）：近窗内被本人撤回的行回进上下文，打
        # 【已改口】标——「记着你的更正」是拟人细节，失忆式抹掉反而吓人；
        # 限条数防挤占正常格（2026-11 改口协议设计，裁决点 2：回）。
        t_anchor = t_now or dh.t
        gone = dh.store.con.execute(
            "SELECT predicate,value,rel FROM fact "
            "WHERE entity=? AND status='retracted' AND valid_to >= ? "
            "ORDER BY valid_to DESC LIMIT ?",
            (ent, t_anchor - config.RETRACTED_CONTEXT_DAYS * 86400.0,
             config.RETRACTED_CONTEXT_MAX)).fetchall()
        for r in gone:
            _add_fact(ent, r["predicate"], r["value"], "retracted", r["rel"])

    # 精确路的 id 直接当图扩散种子（不是靠语义路反推）
    for ent in ents:
        rows = dh.store.con.execute(
            "SELECT fact_id FROM fact WHERE entity=? AND status='active' "
            "ORDER BY written_t DESC LIMIT ?", (ent, k)).fetchall()
        ctx["seeds"].extend(int(r["fact_id"]) for r in rows)

    if not question:
        ctx["seeds"] = list(dict.fromkeys(ctx["seeds"]))
        return ctx

    # ---- B 语义路（唯一的向量召回，全流程只此一次）
    # mood 与检索宽度都从此刻的情绪取：情绪一致检索这一路在 hybrid_score 里
    # 占 0.2 权重，之前传死 0.0 时 `emos = 1 + bias*0*valence` 全等于 1，
    # z 标准化后全 0 —— 那 0.2 是死的（实测 mood=0 与 mood=-0.5 的三路
    # 得分对比：只有 mood≠0 时 emo 项才真的参与排序）。现在接上 P。
    from .limbic.disposition import disposition
    disp = disposition(dh)
    mood = float(disp.mood)
    k_b = disp.scaled_k(k)             # 激动时放宽筐、平静时收窄
    # 取侧性格杠杆（用户提案「存取对记忆加权」）：特质 N 进情节池的遗忘
    # 曲线——敏感的人忘不掉伤心事、皮实的人翻篇快（retrieve._pool_emo）
    g = getattr(dh, "growth", None)
    n_trait = g.traits.get()["N"] if g is not None else None
    try:
        hits = dh.retriever.recall(query_text=question, mood=mood,
                                   t_now=t_now, k_gate=k_b, n_trait=n_trait)
    except Exception:                      # 检索坏了不能拖垮对话
        hits = []
    ctx["hits"] = list(hits)
    ctx["mood"] = mood
    ctx["k_gate"] = k_b
    ctx["disp"] = disp

    seen_spoken = set()
    # 事实节门槛用**真余弦**：recall 返回的是 z 标准化混合分（时间路的正解
    # 常为负数），旧代码拿它当余弦卡 0.5，把时间路捞上来的正解整个误杀——
    # 实测「我多少岁」的两条年龄 fact 混合分 -0.21/-0.79 全被卡掉，LLM 只
    # 看到两句原话。这里按 hit 的真余弦重新卡门槛（阈值语义终于名副其实）；
    # 问句向量 make_embed_fn 按文本缓存，recall 已算过，这里零成本复用。
    qvec = None
    if question and getattr(dh.retriever, "embed_fn", None) is not None:
        try:
            qvec = dh.retriever.embed_fn(question)
        except Exception:                      # noqa: BLE001
            qvec = None
    for kind, rid, _sc in hits:
        if kind == "obs":
            r = dh.store.obs_row(int(rid))
            if r is None or not r["payload"]:
                continue
            said = r["payload"]
            # 本轮这句问话自己刚写进 obs，向量自然和自己最像——不排除的话
            # 「原话记录」第一节永远是用户刚说的话，白占一格还给 LLM 错误锚点
            if question and said == question:
                continue
            if said in seen_spoken:
                continue
            seen_spoken.add(said)
            # 原话必须截到 k 条：它是按情绪与语义混合排出来的，
            # 取全池等于把「情绪决定想起什么」这件事作废——实测 5 条原话
            # 全塞进提示词时，四种情绪下前面的顺序变了，但因为后面几条
            # 原样跟着，读起来完全一样。
            if len(ctx["spoken"]) >= config.RECALL_SPOKEN_K:
                continue
            ctx["spoken"].append((r["source"] or "有人", said))
            continue
        if kind != "fact":
            continue
        r = dh.store.fact_row(int(rid))
        if r is None or r["status"] == "retracted":
            continue
        if qvec is not None:
            from .memory.embed import cosine
            ftext = "%s %s %s" % (r["entity"], r["predicate"], r["value"])
            if cosine(qvec, dh.retriever.embed_fn(ftext)) < config.RECALL_FACT_MIN_COS:
                continue
        _add_fact(r["entity"], r["predicate"], r["value"], r["status"], r["rel"])
        ctx["seeds"].append(int(rid))

    ctx["seeds"] = list(dict.fromkeys(ctx["seeds"]))

    # ---- C 图路
    if config.RECALL_GRAPH_ON and ctx["seeds"]:
        try:
            ctx["related"] = dh.retriever.related(
                ctx["seeds"][:config.RECALL_GRAPH_SEEDS],
                k=config.RECALL_GRAPH_RELATED,
                exclude_entities=set(ents) | {"小星", "我"})
        except Exception:
            ctx["related"] = []
    return ctx


def render_memory_block(dh, ctx):
    """把 ctx 排成给 LLM 的记忆上下文。纯格式化：不查库（只按 id 取行）。

    排版上必须分成三节，因为它们**可信度不同**：
        事实   我提炼出来的 → 可以当答案
        原话   他当时就是这么说的 → 可能有否定/条件/玩笑，不能当结论
        联想   顺着话题想到的 → 不是对方问的事
    混成一节，LLM 会把原话当结论、把联想当答案。
    """
    lines = []
    for ent, pred, val, status, rel in ctx["facts"]:
        tag = config.REL_LABEL.get(rel, "?")
        if status == "retracted":
            # 与「已失效」（superseded/被新值顶掉）区分：改口是用户本人的
            # 更正，LLM 应当知道并遵守，不许把旧行当现行事实复述
            lines.append("【%s·已改口】%s的%s曾是%s（后来本人撤回了）"
                         % (tag, ent, pred, val))
        elif status != "active":
            lines.append("【%s·已失效】%s的%s曾是%s" % (tag, ent, pred, val))
        else:
            lines.append("【%s】%s的%s是%s" % (tag, ent, pred, val))

    if ctx["spoken"]:
        lines.append("")
        lines.append("【原话记录】（当时说过的原话，可能含条件/否定/语气，"
                     "引用时照原话说，别当成已确认的事实）")
        for who, said in ctx["spoken"]:
            lines.append("【%s】「%s」" % (who, said))

    if ctx["related"]:
        lines.append("")
        lines.append("【这让我又想起来】（顺着上面的话题想到的，别当成对方问的事）")
        for fid, _a in ctx["related"]:
            r = dh.store.fact_row(int(fid))
            if r is None:
                continue
            tag = config.REL_LABEL.get(int(r["rel"] or 0), "?")
            if r["status"] != "active":
                lines.append("【%s·已失效】%s的%s曾是%s"
                             % (tag, r["entity"], r["predicate"], r["value"]))
            else:
                lines.append("【%s】%s的%s是%s"
                             % (tag, r["entity"], r["predicate"], r["value"]))

    if not lines:
        return "（记忆里还没有相关的事。）"
    return "\n".join(lines)
