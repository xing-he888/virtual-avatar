# -*- coding: utf-8 -*-
"""写回步（七步循环第 6 步）：把一次经历写进五个区。

纪律 [DESIGN §2 / STORE §2 / GRAPH §10.1]：
    R1 情节流只追加，永不修改——改记忆是在语义网上加新版本，不是回头改旧记录。
    R2 每条事实带有效期：一行 = 一个值的一段有效期。
    R3 事实记忆**一律全写**（FIFO），不按唤醒度挑着写——E3/E5/E7 三次否证。
       唤醒度门控只给情节/事件记忆用。
    R4 来源是边上的一列属性，不是可遍历的节点（张冠李戴的修法）。
    R5 语义向量只给 fact 和 mention 算，不给每条 obs 算 [STORE §5.3 第 3 条]。
"""
from collections import defaultdict

from . import config, kernel
from .calib import StreamStats
from .extract import Extractor, detect_relations, is_segment_boundary
from . import extract

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
        self.event_window = event_window
        self.clock = clock
        # 标定：每个 (实体|谓词) 一条观测流
        self.streams = defaultdict(StreamStats)
        self._prev_t = None          # 段边界检测
        self._open_event = {}        # key -> event_id

    # ------------------------------------------------ 主入口
    def hear(self, text, t, speaker="用户", channel="text", valence=None):
        """用户/他人说了一句话。返回写入摘要 dict。

        来源规则：obs.source 记「这信息最早是谁说的」——带「听X说」前缀时
        是 X 不是说话人 [GRAPH §10.1]；来源留最早那条 [extract.py 注释 2]。
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
                                    valence=valence)
        self._prev_t = t
        out = dict(obs_id=obs_id, seg=seg, facts=[], mention=None, event=None)
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
        fid, action = self.store.write_fact(
            subject, pred, val, t, obs_id, rel=rel,
            conf=0.9 if rel == 1 else 0.75, force_new=parsed["change"],
            revive_ok=revive_ok)
        out["facts"].append(dict(fact_id=fid, action=action, key=key, value=val,
                                 src=src_name))
        # 标定观测流：把这次「看到的值」喂进去
        self.streams[key].observe(val)
        # 事件：同 (主体|谓词) 时间窗内挂同一块牌子
        ev = self._attach_event(key, t, obs_id, text)
        out["event"] = ev
        # 语义向量只给 fact 和 mention 算（R5）
        self.store.put_vec("fact", fid, self.embed_fn("%s %s %s" % (subject, pred, val)))
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
        self.store.put_vec("mention", mid, self.embed_fn(text))
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
            fid, action = self.store.write_fact(
                c.subject, c.predicate, c.value, t, obs_id, rel=c_rel,
                conf=min(0.95, max(0.5, c.conf)),
                force_new=self.ex.signals_change(text), revive_ok=revive_ok)
            key = "%s|%s" % (c.subject, c.predicate)
            # 与 hear 相同的三个后续：标定流、事件牌子、语义向量
            self.streams[key].observe(c.value)
            ev = self._attach_event(key, t, obs_id, text)
            self.store.put_vec("fact", fid,
                               self.embed_fn("%s %s %s" % (c.subject, c.predicate,
                                                           c.value)))
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
