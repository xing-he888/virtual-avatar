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
from collections import defaultdict

from .. import config
from . import embed as embed_mod
from . import kernel
from .calib import StreamStats
from .equiv import ValueEquivalence
from ..extract import Extractor, detect_relations, is_segment_boundary
from .. import extract

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
        冲突关系自动撤回。返回 (事实列表, 被撤回的旧关系列表)。"""
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
                conf=0.9 if rel == 1 else 0.75, force_new=True, revive_ok=True)
            facts.append(dict(fact_id=fid, action=action,
                              key="%s|%s" % (frm, pred), value=to, src=None))
        return facts, retracted

    def _deref_pronouns(self, cands):
        """写入前的代词防护：
        「我」在说话人自己的句子里确定性地指属主（meta 表登记）——
        映射过去；没登记属主时按无锚指代处理，拒写。
        第三人称/集合指代无法安全消解，整条丢弃（宁可漏记 [DESIGN §6]）。
        """
        owner = self.store.get_owner()
        out = []
        for c in cands:
            if c.subject == "我":
                if not owner:
                    continue
                c.subject = owner
            elif c.subject in self.ANAPHORA_DROP:
                continue
            if c.is_relation:
                if c.value == "我":
                    if not owner:
                        continue
                    c.value = owner
                if c.value in self.ANAPHORA_DROP:
                    continue
            out.append(c)
        return out

    def hear_llm_candidates(self, text, t, speaker, obs_id, cands,
                            learn_entity_cb=None):
        """LLM 抽取的候选写成事实。与 hear 走同一套存储规则
        （单值互斥/多值并存/传旧话不复活/复活须本人宣布）——
        LLM 只负责「指认」，存储层负责伤害控制。

        cands: extract_llm.Candidate 列表（已过五道闸门）。
        learn_entity_cb(name): 新实体回调（别名回写+持久化）。
        返回 dict(fact_ids=[...], actions=[...])。
        """
        cands = self._deref_pronouns(cands)
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
