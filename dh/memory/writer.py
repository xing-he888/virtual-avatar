# -*- coding: utf-8 -*-
"""写回步（七步循环第 6 步）：把一次经历写进五个区。

纪律 [DESIGN §2 / STORE §2 / GRAPH §10.1]：
    R1 情节流只追加，永不修改——改记忆是在语义网上加新版本，不是回头改旧记录。
    R2 每条事实带有效期：一行 = 一个值的一段有效期。
    R3 事实记忆**一律全写**（FIFO），不按唤醒度挑着写——E3/E5/E7 三次否证。
       唤醒度门控只给情节/事件记忆用。
    R4 来源是边上的一列属性，不是可遍历的节点（张冠李戴的修法）。
    R5 语义向量给 **fact + obs** 算；mention 照算但与 obs 同文，不进检索。
       原话向量**不依赖抽取**——「老王搬海淀去了」「老王是我的朋友」这类
       正则抽不出的句子，恰恰只有这条路能把它们召回；挂在 parsed 之后
       等于让 RAG 只在容易的地方工作（实测 5 句里 3 句零向量）。
       [推翻 STORE §5.3 第 3 条「不给每条 obs 算」：那条的前提是 R5 省空间，
       但实测 bge 单句 0.12 ms、向量 512 维，一条 obs 的向量成本低于
       它的 SQLite 行本身；而漏掉原话的代价是整条记忆不可检索。]
"""
import re
from collections import defaultdict

from .. import config
from . import embed as embed_mod
from . import kernel
from .calib import StreamStats
from .equiv import ValueEquivalence
from ..extract import Extractor, detect_relations, is_segment_boundary
from .. import extract

# owner 名字清洗：错抓的词几乎都会撞上这些禁字（「我叫朱宏璟是你的主任」
# 抓「主任」撞「主」不在禁字——所以还叠加长度与代词检查，见 _clean_owner_name）
_OWNER_BLOCK = "的我你他她它您说叫是的"
_OWNER_PRONOUNS = {"我", "你", "咱", "他", "她", "它"}


def _clean_owner_name(value):
    if not value:
        return None
    name = str(value).strip("「」\"'“”。．，,！!？?呀呢哈哦啊吧了呢 ")
    # 尾部裁虚词：「我叫朱宏璟是你的主任」的贪婪捕获给出「朱宏璟是你的」，
    # 是/你/的 都在禁字表——从尾往头剥（真名极少以这些字收尾）
    while name and name[-1] in _OWNER_BLOCK:
        name = name[:-1]
    if not name or not (1 <= len(name) <= 6):
        return None
    if name in _OWNER_PRONOUNS or any(ch in _OWNER_BLOCK for ch in name):
        return None
    return name

# 事实向量：无网络的确定性哈希嵌入（256 维 char 2-gram，L2 归一）。
# 有网时把 embed_fn 换成 DashScope text-embedding-v3（1024 维）即可 [STORE §4.3]。
# 注意必须用稳定哈希（crc32）：内置 hash() 对字符串按进程随机化，
# 会让 sim 边和语义路在不同进程里时有时无——同种子同结果就没了。
VEC_DIM = 256


def hash_embed(text, dim=VEC_DIM):
    import numpy as np
    import zlib
    v = np.zeros(dim, dtype=np.float32)
    s = "".join(ch for ch in text if not ch.isspace())
    for i in range(len(s) - 1):
        v[zlib.crc32(("%s|%s" % (s[i], s[i + 1])).encode("utf-8")) % dim] += 1.0
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


class Writer:
    """写回步。持有一个 MemoryStore、一个 Extractor 和标定用的观测流统计。"""

    def __init__(self, store, alias=None, embed_fn=None, event_window=800.0,
                 clock=lambda: 0.0):
        self.store = store
        self.ex = Extractor(alias)
        self.embed_fn = embed_fn or hash_embed
        # 向量来源标识：写进 vec.model。换了模型必须整表重算——
        # 不同模型的余弦不可比（load_matrix 按 model 过滤）
        self.embed_model = getattr(embed_fn, "model_name", None) or getattr(
            embed_mod, "HASH_NAME")
        self.event_window = event_window
        self.clock = clock
        # 标定：每个 (实体|谓词) 一条观测流
        self.streams = defaultdict(StreamStats)
        # 写入侧值等价裁决（词典→别名→豁免→value-only 余弦，见 equiv.py）
        self.equiv = ValueEquivalence(store, self.embed_fn)
        self._prev_t = None          # 段边界检测
        self._open_event = {}        # key -> event_id

    # ------------------------------------------------ 带等价裁决的写值
    def _write_value(self, entity, predicate, value, t, obs_id, rel, conf,
                     force_new, revive_ok):
        """写值主口：先等价裁决，裁决不下再走 write_fact 的字面四分支。

        变更信号词一票否决合并（force_new=True 时跳过裁决）——说话人在
        宣布变更（搬到/换号/改成…），相似度再高也是新值：
        「搬到海淀区北苑」与「海淀区」余弦 0.837，但这是搬家不是复述。

        返回 (fact_id, action, canonical)。confirm_equiv 时 fact.value 保持
        先到的规范值不动，新表面形式只进 value_alias 和 obs 原文。"""
        eq = None
        if not force_new:
            try:
                eq = self.equiv.find(entity, predicate, value)
            except Exception:        # noqa: BLE001  裁决坏了回到旧行为，不挡写入
                eq = None
        if eq is not None:
            fid, canonical = eq
            self.store.confirm_fact(fid, t)
            if value != canonical:
                self.store.value_alias_learn(entity, predicate, value,
                                             canonical, t=t)
            return fid, "confirm_equiv", canonical
        fid, action = self.store.write_fact(
            entity, predicate, value, t, obs_id, rel=rel, conf=conf,
            force_new=force_new, revive_ok=revive_ok)
        return fid, action, value

    # ------------------------------------------------ 主入口
    def hear(self, text, t, speaker="用户", channel="text", valence=None,
             salience=0.5):
        """用户/他人说了一句话。返回写入摘要 dict。

        来源规则：obs.source 记「这信息最早是谁说的」——带「听X说」前缀时
        是 X 不是说话人 [GRAPH §10.1]；来源留最早那条 [extract.py 注释 2]。
        salience：右脑的情绪编码强度（tick 第 6 步逐条算好传入）；决定这段
        情节记得多深，不决定行存不存在。
        """
        prev_t = self._prev_t
        seg = is_segment_boundary(t, prev_t)
        parsed = self.ex.parse(text, speaker=speaker)
        src_name = speaker
        rel = 1 if speaker in ("用户", "我", "小星") else 2   # 用户亲口=L1，他人转述=L2
        if parsed is not None and parsed["src"] is not None:
            src_name = parsed["src"]                          # 剥出来的传话人才是来源
            rel = 2
        # 本条经历涉及谁 = 说话人（若能归名：用户→属主，其余按别名词典）
        # + 文中提到的实体。1:1 聊天里双方名字是元数据不在正文里，
        # 说话人并入才有共现原料——识海 co 边（同话兜底边）靠它
        _, body = self.ex.split_source(text)
        owner = self.store.get_owner()
        speaker_ent = (owner if speaker in ("用户", "我")
                       else self.ex.canon(speaker))
        ents_all = ",".join(dict.fromkeys(
            ([speaker_ent] if speaker_ent else [])
            + [e[2] for e in self.ex.find_entities(body)])) or None
        obs_id = self.store.add_obs(t, "told", channel, text, entity=ents_all,
                                    source=src_name,
                                    rel=rel, seg_bnd=1.0 if seg else 0.0,
                                    valence=valence, salience=salience)
        self._prev_t = t
        out = dict(obs_id=obs_id, seg=seg, facts=[], mention=None, event=None)
        # 原话向量（kind=obs，row_id=obs_id）：**在抽取之前、无条件**写。
        # 它只依赖 (obs_id, text)，跟 parsed 死活无关；挂在 par.parsed 之后
        # 会让「正则抽不出的句子」同时失去事实和向量——两条路一起瞎。
        # 这条也是 RAG 的唯一入口：问「老王住哪儿」时，「老王搬海淀去了」
        # 这句没有任何 fact 行可点查，只能靠向量捞回来。
        self.store.put_vec("obs", obs_id, self.embed_fn(text),
                           model=self.embed_model)
        # 关系事实（Tier 0 正则）：「小王是小李他哥」这类句子抽不出常规
        # 三元组，但关系本身就是事实——写回 + 冲突自动撤回
        rel_facts, rel_retracted = self._hear_relations(body, t, obs_id, rel)
        out["facts"].extend(rel_facts)
        if rel_retracted:
            out["relations_retracted"] = rel_retracted
        if parsed is None:
            return out
        subject, pred, val = parsed["subject"], parsed["predicate"], parsed["value"]
        key = "%s|%s" % (subject, pred)
        # 复活权：只有本人亲口宣布变更才允许把已失效的值改回来
        #（「我换号了」；传话人/旁人说「换号了」可能是传旧话）
        revive_ok = (speaker == subject)
        fid, action, canonical = self._write_value(
            subject, pred, val, t, obs_id, rel=rel,
            conf=0.9 if rel == 1 else 0.75, force_new=parsed["change"],
            revive_ok=revive_ok)
        out["facts"].append(dict(fact_id=fid, action=action, key=key,
                                 value=canonical, src=src_name))
        # 标定观测流：喂规范值——表面形式的搅动不再计入翻转率（phi 干净了）
        self.streams[key].observe(canonical)
        # 事件：同 (主体|谓词) 时间窗内挂同一块牌子
        ev = self._attach_event(key, t, obs_id, text)
        out["event"] = ev
        # 语义向量只给 fact 和 mention 算（R5）；confirm_equiv 是同一行同一文本，
        # upsert 幂等，无额外成本
        self.store.put_vec("fact", fid,
                           self.embed_fn("%s %s %s" % (subject, pred, canonical)),
                           model=self.embed_model)
        return out

    def _attach_event(self, key, t, obs_id, text):
        ev = self.store.latest_open_event(key, t, self.event_window)
        if ev is None:
            label = key + " 变动"
            ev = self.store.open_event(label, t, key=key)
            self._open_event[key] = ev
        else:
            old = self._open_event.get(key)
            if old is not None and old != ev:
                self.store.close_event(old, t)
            self._open_event[key] = ev
        mid = self.store.add_mention(ev, obs_id, text)
        # 不再给 mention 算向量：它的 text 与上面那条 obs 一模一样，
        # 同模型同输入算出来是同一个向量，存两份 = 双倍存储 + 双倍索引，
        # 而 row_id 是 mention_id（与 obs_id 不同空间），检索取时间/效价
        # 还得回表翻译一次——纯粹是给自己挖坑。原话向量统一挂 kind="obs"。
        return dict(event_id=ev, mention_id=mid)

    # ------------------------------------------------ LLM 补记
    # 第三人称/集合指代在没有上下文消解时写谁都是错的——宁可漏记 [DESIGN §6]
    ANAPHORA_DROP = {"他", "她", "它", "我们", "你们", "他们", "她们", "它们", "咱", "大家"}

    def _learn_new_entity(self, name):
        """正则路径的新实体学习：别名词典 + 持久化（与 LLM 补记同款闭环）。"""
        if self.ex.learn_alias(name, name):
            self.store.alias_learn(name, name)
        return name

    def _retract_conflicting_relations(self, frm, rel_word, to, t):
        """同一对人之间互斥的关系（哥哥 vs 弟弟）：新关系落位时把旧的撤回。

        互斥是关系本身的语义——记错关系=换关系，不是多值并存。
        返回 [(旧关系词, 对方)] 供回复确认用。
        """
        out = []
        for other in extract.REL_CONFLICT.get(rel_word, ()):
            rows = self.store.con.execute(
                "SELECT fact_id FROM fact WHERE entity=? AND predicate=? "
                "AND value=? AND status='active'",
                (frm, "关系." + other, to)).fetchall()
            for r in rows:
                if self.store.retract_fact(r["fact_id"], t):
                    out.append((other, to))
        return out

    def _hear_relations(self, body, t, obs_id, rel):
        """正则抽出的关系写成事实（Tier 0）。名字不在别名词典的当场学习；
        冲突关系自动撤回。返回 (事实列表, 被撤回的旧关系列表)。

        revive_ok 收紧到「本人宣布」口径：说话人就是关系主体时才许复活
        （复述自己的关系=改回旧身份）；L2 传话人替别人复活是传旧话纪律
        的违规（2026-11 审查 C4 顺手修）。"""
        facts, retracted = [], []
        for frm, rel_word, to in detect_relations(body):
            frm = self.ex.canon(frm) or self._learn_new_entity(frm)
            to = self.ex.canon(to) or self._learn_new_entity(to)
            if not frm or not to:
                continue
            pred = "关系." + rel_word
            self.store.declare_pred(pred, multi=1)
            retracted.extend(
                self._retract_conflicting_relations(frm, rel_word, to, t))
            fid, action = self.store.write_fact(
                frm, pred, to, t, obs_id, rel=rel,
                conf=0.9 if rel == 1 else 0.75, force_new=True,
                revive_ok=(rel == 1))
            facts.append(dict(fact_id=fid, action=action,
                              key="%s|%s" % (frm, pred), value=to, src=None))
        return facts, retracted

    def _deref_pronouns(self, cands, speaker="用户"):
        """写入前的代词防护：
        「我」在说话人自己的句子里确定性地指属主（meta 表登记）——
        映射过去；没登记属主时按无锚指代处理，拒写。
        「你」在用户的话里确定性指小星（1:1 对话，对方的锚是常量）——
        旧版漏了这个方向，「我是你的主任」的 L1 关系候选写成
        你|关系.主任=朱宏璟（实体「你」是代词不是人），点查永远不可达
        （2026-11 改口协议审查 H4）。
        第三人称/集合指代无法安全消解，整条丢弃（宁可漏记 [DESIGN §6]）。
        """
        owner = self.store.get_owner()
        out = []
        for c in cands:
            if c.subject == "我":
                if not owner:
                    continue
                c.subject = owner
            elif speaker != "小星" and c.subject == "你":
                c.subject = "小星"
            elif c.subject in self.ANAPHORA_DROP:
                continue
            if c.is_relation:
                if c.value == "我":
                    if not owner:
                        continue
                    c.value = owner
                elif speaker != "小星" and c.value == "你":
                    c.value = "小星"
                if c.value in self.ANAPHORA_DROP:
                    continue
            out.append(c)
        return out

    def maybe_register_owner_from_candidates(self, cands, speaker):
        """owner 自愈（Tier 1 姓名候选）：owner 未登记且用户亲口报了名
        （LLM 抽成 姓名/名字 谓词）→ 当场登记。

        旧版只有「我叫X」整句收尾的正则（_SELF_INTRO_RE），句子带任何
        后缀就废——「我叫朱宏璟是你的主任」永远登不上 owner，代词防护
        随之把「我」候选全部静默丢弃（2026-11 审查 H1/H3，「主人」的
        根因之一）。守卫：只认姓名谓词、值非代词/禁字、speaker 非小星、
        已有 owner 不动（不许改名——改名走修正面板）。"""
        if speaker == "小星" or self.store.get_owner():
            return None
        for c in cands:
            if getattr(c, "predicate", "") not in ("姓名", "名字"):
                continue
            name = _clean_owner_name(getattr(c, "value", None))
            if name:
                self.store.set_owner(name, note="llm-name")
                try:
                    self.store.alias_learn(name, name)
                except Exception:                    # noqa: BLE001
                    pass
                return name
        return None

    def register_owner_from_history(self, speaker="用户"):
        """改口协议的属主兜底：owner 未登记时，从说话人历史原话里找
        「我叫X」补登记（只读扫描 + 现有 set_owner，不造新事实）。
        真实库实证：obs 22「我叫朱宏璟是你的主任」在，owner 却是空。"""
        cur = self.store.get_owner()
        if cur:
            return cur
        rows = self.store.con.execute(
            "SELECT payload FROM obs WHERE source=? AND payload LIKE '我叫%' "
            "ORDER BY t DESC LIMIT 20", (speaker,)).fetchall()
        for r in rows:
            m = re.search(r"我(?:叫|叫做|名为)([\u4e00-\u9fffA-Za-z0-9]{1,6})",
                          r["payload"])
            name = _clean_owner_name(m.group(1) if m else None)
            if name:
                self.store.set_owner(name, note="history-intro")
                try:
                    self.store.alias_learn(name, name)
                except Exception:                    # noqa: BLE001
                    pass
                return name
        return None

    # ------------------------------------------------ 本人改口协议（执行层）
    def _grounded_in_history(self, speaker, term):
        """锚定校验：锚词必须逐字出现在**本说话人**的历史原话里。

        LIKE 而非 obs_fts MATCH：默认分词器不切 CJK，MATCH「主任」实测
        命中 0（2026-11 审查 H9）。这是防洗库的第一道闸——LLM 指认的
        锚词、L0 捕获的锚词，落不到发起人自己的原话里就一个字不动。"""
        if not term:
            return False
        row = self.store.con.execute(
            "SELECT 1 FROM obs WHERE source=? AND payload LIKE ? LIMIT 1",
            (speaker, "%" + term + "%")).fetchone()
        return row is not None

    def retract_by_anchor(self, speaker, term, t):
        """词级撤回：来源 × 锚双重限定。

        撤「说话人自己说出的、主张过 term 的」active 事实：
        predicate=关系.term（身份槽），或 value=term（值槽，谓词不限——
        「他住院的事我说错了」要能撤病情；值=锚 意味着说话人断言过
        「某物是锚」，撤回主张即全部作废。误伤面由两道闸兜：来源守卫
        （只撤自己的话）+ 锚定校验（锚必须逐字在自己的历史原话里））。
        **不按 src_obs 全量连坐**：一句原话派生多条事实时用户通常只撤
        其中一条（连坐是句级作废形态的事，见 retract_by_utterance）。"""
        rows = self.store.con.execute(
            "SELECT f.fact_id, f.entity, f.predicate, f.value FROM fact f "
            "JOIN obs o ON o.obs_id=f.src_obs WHERE f.status='active' "
            "AND o.source=? AND (f.predicate=? OR f.value=?)",
            (speaker, "关系." + term, term)).fetchall()
        out = []
        for r in rows:
            if self.store.retract_fact(r["fact_id"], t):
                out.append(dict(fact_id=r["fact_id"], entity=r["entity"],
                                predicate=r["predicate"], value=r["value"]))
        return out

    def _retract_facts_of_obs(self, obs_id, t):
        out = []
        for r in self.store.con.execute(
                "SELECT fact_id, entity, predicate, value FROM fact "
                "WHERE src_obs=? AND status='active'", (obs_id,)).fetchall():
            if self.store.retract_fact(r["fact_id"], t):
                out.append(dict(fact_id=r["fact_id"], entity=r["entity"],
                                predicate=r["predicate"], value=r["value"]))
        return out

    def retract_by_utterance(self, speaker, quote, t):
        """句级作废：quote 是说话人某句原话的逐字子串 → 该句派生的全部
        active fact 连坐（「我那句话作废」的形态）。取最近 3 条候选里
        最新的一条命中。"""
        if not quote or not self._grounded_in_history(speaker, quote[:16]):
            return []
        row = self.store.con.execute(
            "SELECT obs_id FROM obs WHERE source=? AND payload LIKE ? "
            "ORDER BY t DESC LIMIT 1", (speaker, "%" + quote + "%")).fetchone()
        return self._retract_facts_of_obs(row["obs_id"], t) if row else []

    def apply_correction(self, speaker, hit, t, text, obs_id):
        """改口协议编排：LLM/L0 只指认，执行全在这里（确定性、可 review）。

        hit: rules.RetractionHit（L0 直出，或 L1 指认经校验后的形态）；
        纯自陈身份（无撤回）也走这里（hit=RetractionHit(new_value=role)）。
        返回 dict(retracted, written, grounded)——grounded=False = 锚没落
        到说话人历史里，**一个字都不动**，上层诚实回话。
        撤回按「来源×锚」；新主张写 关系.R（方向：说话人 → 小星），
        force_new+revive_ok=True——本人当场宣布的主张就是「复活须本人
        宣布」的那个本人（改回旧身份同样走这里，复述即复活）。
        留痕：channel='correct' 的一行 obs（成长层「被纠正+信任」的
        证据 SQL 认的就是 channel，2026-11 审查 C4 顺手修的口径）。"""
        retracted, written = [], []
        owner = self.register_owner_from_history(speaker)
        grounded = False
        if hit is not None and (hit.term or hit.whole_sentence):
            if hit.term and self._grounded_in_history(speaker, hit.term):
                grounded = True
                retracted = self.retract_by_anchor(speaker, hit.term, t)
            elif hit.whole_sentence:
                quote = getattr(hit, "quote", None)
                if quote:
                    retracted = self.retract_by_utterance(speaker, quote, t)
                    grounded = bool(retracted)
                else:
                    # 无引文的句级作废：撤回说话人最近一句原话的全部产出
                    row = self.store.con.execute(
                        "SELECT obs_id, payload FROM obs WHERE source=? "
                        "AND obs_id != ? ORDER BY t DESC LIMIT 1",
                        (speaker, obs_id)).fetchone()
                    if row is not None:
                        retracted = self._retract_facts_of_obs(row["obs_id"], t)
                        grounded = bool(retracted)
        role = (hit.new_value if hit is not None else None)
        if role:
            pred = "关系." + role
            self.store.declare_pred(pred, multi=1)
            subject = owner or ("用户" if speaker in ("用户", "我") else speaker)
            fid, action, canonical = self._write_value(
                subject, pred, "小星", t, obs_id, rel=1, conf=0.95,
                force_new=True, revive_ok=True)
            written.append(dict(fact_id=fid, action=action, entity=subject,
                                predicate=pred, value=canonical))
            grounded = True
        if retracted or written:
            self.store.add_obs(t, "told", "correct",
                               "【本人改口】" + text[:60], source=speaker,
                               rel=1, seg_bnd=0.0)
        return dict(retracted=retracted, written=written, grounded=grounded)

    def hear_llm_candidates(self, text, t, speaker, obs_id, cands,
                            learn_entity_cb=None):
        """LLM 抽取的候选写成事实。与 hear 走同一套存储规则
        （单值互斥/多值并存/传旧话不复活/复活须本人宣布）——
        LLM 只负责「指认」，存储层负责伤害控制。

        cands: extract_llm.Candidate 列表（已过五道闸门）。
        learn_entity_cb(name): 新实体回调（别名回写+持久化）。
        返回 dict(fact_ids=[...], actions=[...])。
        """
        # owner 自愈（副作用调用）：登记成功与否都继续走原候选流
        self.maybe_register_owner_from_candidates(cands, speaker)
        cands = self._deref_pronouns(cands, speaker)
        fact_ids, actions = [], []
        # 来源分级与 hear 一致：剥出「听X说」=L2；用户/自己亲口=L1
        parsed_src = self.ex.split_source(text)[0]
        rel = 2 if parsed_src is not None else (
            1 if speaker in ("用户", "我", "小星") else 2)
        force_new = self.ex.signals_change(text)     # 变更信号：一票否决等价合并
        for c in cands:
            c_rel = 2 if c.tense == "hearsay" else rel   # 传闻一律 L2
            if c.is_relation:
                c.multi = True                            # 关系谓词恒多值
                self._retract_conflicting_relations(      # 记错关系=换关系
                    c.subject, c.predicate.split(".", 1)[-1], c.value, t)
            if learn_entity_cb is not None and c.is_new_entity:
                learn_entity_cb(c.subject)
            # 自定义谓词注册进存储层
            if c.is_new_predicate or c.is_relation:
                self.store.declare_pred(c.predicate, multi=int(c.multi))
            revive_ok = (speaker == c.subject)
            fid, action, canonical = self._write_value(
                c.subject, c.predicate, c.value, t, obs_id, rel=c_rel,
                conf=min(0.95, max(0.5, c.conf)),
                force_new=force_new, revive_ok=revive_ok)
            key = "%s|%s" % (c.subject, c.predicate)
            # 与 hear 相同的三个后续：标定流（规范值）、事件牌子、语义向量
            self.streams[key].observe(canonical)
            ev = self._attach_event(key, t, obs_id, text)
            self.store.put_vec("fact", fid,
                               self.embed_fn("%s %s %s" % (c.subject, c.predicate,
                                                           canonical)),
                               model=self.embed_model)
            fact_ids.append(fid)
            actions.append(action)
        return dict(fact_ids=fact_ids, actions=actions)

    # ------------------------------------------------ 标定接线
    def maybe_calibrate(self, schedule, t):
        """攒够观测就写一次标定行；冻结由 schedule 管。"""
        best = None
        for key, st in self.streams.items():
            s, feats, wrote = schedule.observe(st)
            if wrote:
                self.store.calib_set(schedule.scope, key, "n", st.n, note=feats,
                                     fitted_t=t)
            if s is not None:
                best = (s, feats)
        return best
