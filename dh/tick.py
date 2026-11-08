# -*- coding: utf-8 -*-
"""七步 tick 循环 [DESIGN §1]。顺序不能乱——后面每一步都读前一步留下的东西。

    1 感知    输入变成带时间戳的观测
    2 注意    容量是硬的，进不来的丢（留在感觉缓冲里，2 秒后消失）
    3 检索    从记忆里捞出跟当前情境相关的几条
    4 评估    这事对我意味着什么？更新情绪状态
    5 反应    想 / 说 / 做什么都不做
    6 写回    把这次经历写进记忆
    7 离线    空闲时重放、巩固、心智游移（idle()）

它不是「会说话的问答框」，是每 tick 做同样七件事、其中六件跟你的问题
无关的心智循环 [DESIGN §0]。
"""
import numpy as np

from . import config
from .calib import CalibSchedule
from .conflict import ConflictDetector, SlowSystem
from .emotion import AffectSystem, Appraisal, appraise_to_pad
from .extract import Extractor
from .graph_layer import ActivationGraph
from .offline import Offline
from .retrieve import Answer, Retriever
from .store import MemoryStore
from .writer import Writer

QUESTION_WORDS = ("吗", "呢", "？", "?", "多少", "什么", "几", "怎么")


def looks_like_question(text):
    return any(w in text for w in QUESTION_WORDS)


class DigitalHuman:
    """事件驱动的七步循环：hear()/ask() 触发一轮 tick，idle() 触发第 7 步。"""

    def __init__(self, store=None, alias=None, embed_fn=None, seed=0,
                 S=None, sigma=None, debug=False):
        from .writer import hash_embed
        embed_fn = embed_fn or hash_embed     # 语义路默认离线哈希嵌入
        self.store = store or MemoryStore()
        self.writer = Writer(self.store, alias=alias, embed_fn=embed_fn)
        self.retriever = Retriever(self.store, embed_fn=embed_fn, S=S, sigma=sigma)
        self.graph = ActivationGraph(self.store)
        self.affect = AffectSystem()
        self.affect.rng = np.random.default_rng(seed + 1)   # 情绪噪声也进种子
        from .working_mem import WorkingMemory
        self.wm = WorkingMemory()
        self.conflict = ConflictDetector(self.store)
        self.slow = SlowSystem(self.store, self.retriever)
        self.offline = Offline(self)
        self.ex = self.writer.ex
        self.schedule = CalibSchedule(self.store)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.sensory = []            # 感觉缓冲（约 2 秒 TTL）
        self._drive = None           # 本 tick 内被接受的情绪冲击（速行窗口）
        self.idle_since = None
        self.debug = debug
        self.log = []
        # 谓词注册：多值必须是谓词自带的属性，不能全局设阈值 [STORE §2]
        for pred, multi in (("电话", 0), ("备用电话", 1), ("工作单位", 0),
                            ("科室", 0), ("住址", 0), ("病情", 0)):
            self.store.declare_pred(pred, multi)
        # 别名词典：LLM 历史发现的新实体加载进正则快路（自学习的持久半边）
        for r in self.store.alias_all():
            self.writer.ex.learn_alias(r["name"], r["canon"])

    # ------------------------------------------------ 对外入口
    def hear(self, text, speaker="用户", t=None, valence=None):
        """有人说了句话。跑一轮完整 tick（第 5 步多半是「什么都不做」）。"""
        t = self.t if t is None else float(t)
        self.t = max(self.t, t)
        if valence is None:
            valence = Extractor.valence(text)
        self.sensory.append(dict(text=text, speaker=speaker, t=t, valence=valence,
                                 is_question=looks_like_question(text)))
        return self.tick()

    def ask(self, text, t=None):
        """问它一句话：检索 -> 评估 -> 反应。返回 (Answer, 措辞, 诊断)。"""
        t = self.t if t is None else float(t)
        self.t = max(self.t, t)
        rec = dict(t=t, kind="ask", text=text)
        # 3 检索：问题 -> (实体, 谓词) -> 点查（微秒级 [STORE §5.2]）
        parsed = self.ex.parse_question(text) or self.ex.parse(text)
        diag = dict(parsed=parsed)
        if parsed is not None:
            ans = self.retriever.ask_fact(parsed["subject"], parsed["predicate"],
                                          asof=t)
        else:
            ans = Answer([], abstain=True)
        # 4 评估：被问到某人的事本身不引入情绪事件
        self._appraise(valence=0.0, novelty=0.1)
        self.affect.tick(1000.0, drive=self._drive)
        self._drive = None
        # 5 反应：冲突检测 -> 慢系统（默认不启动，被冲突唤醒 [DESIGN §5.1]）
        flags = self.conflict.check(question_text=text, answer=ans,
                                    arousal=self.affect.arousal(),
                                    valence=self.affect.state.p)
        diag["conflict_flags"] = flags
        if flags and parsed is not None:
            self.slow.reset()
            diag["slow"] = self.slow.deliberate(parsed["subject"],
                                                parsed["predicate"],
                                                question_text=text)
        # 6 写回：问题本身也是一次经历（问句不断言，不写事实）
        self.store.add_obs(t, "told", "text", text, source="用户", rel=1, seg_bnd=0.0)
        outcome = "abstain" if ans.abstain else ("degraded" if ans.degraded else "ok")
        self.store.log_qa(text, t, outcome)
        rec["answer"] = ans.phrase()
        self.log.append(rec)
        return ans, ans.phrase(), diag

    def idle(self, t=None, rng=None):
        """第 7 步：巩固 + 心智游移。"""
        t = self.t if t is None else float(t)
        rng = rng or self.rng
        narr = self.offline.consolidate(t)
        wander = self.offline.wander(t, rng)
        self.idle_since = None
        return dict(narratives=narr, wander=wander)

    # ------------------------------------------------ 七步
    def tick(self, dt_s=1.0):
        t = self.t
        rec = dict(t=t, kind="tick")
        # 1 感知：感觉缓冲 TTL 约 2 秒，过期的直接消失 [DESIGN §2]
        alive = [x for x in self.sensory if t - x["t"] <= config.SENSORY_TTL_S]
        self.sensory = alive
        # 2 注意：容量硬上限，进不来的留在缓冲里等下一 tick
        batch = alive[:config.ATTEND_PER_TICK]
        attended = []
        for x in batch:
            sal = self.wm.salience(x.get("valence") or 0.0, self.affect.arousal())
            attended.append(dict(obs_id=None, text=x["text"], salience=sal,
                                 t=x["t"], speaker=x["speaker"],
                                 valence=x.get("valence"),
                                 is_question=x["is_question"]))
        self.sensory = alive[len(batch):]
        rec["attended"] = len(attended)
        # 3 检索 + 4 评估（逐条进工作记忆；4 槽硬上限，挤掉显著性最低的）
        for item in attended:
            self.wm.push(item)
            self._appraise(valence=item.get("valence") or 0.0,
                           novelty=0.3 if item.get("is_question") else 0.1)
            self.affect.tick(dt_s * 1000.0, drive=self._drive)  # 收毫秒
            self._drive = None
        # 5 反应：陈述不触发说话（说话由 ask()/上层驱动）；这里只记状态
        rec["mood"] = dict(p=self.affect.state.p, a=self.affect.state.a,
                           d=self.affect.state.d)
        # 6 写回：把经历写进记忆（事实记忆一律全写，不按唤醒度挑 [DESIGN §4.3 补注]）
        rec["written"] = []
        rec["obs_ids"] = []
        rec["relation_facts"] = []          # 本轮写的关系明细（动态修正确认用）
        rec["relations_retracted"] = []     # 本轮被撤回的冲突旧关系
        for item in attended:
            if item.get("is_question"):
                continue
            out = self.writer.hear(item["text"], item["t"], item["speaker"],
                                   valence=item.get("valence"))
            rec["obs_ids"].append(out["obs_id"])
            if out["facts"]:
                fid = out["facts"][0]["fact_id"]
                rec["written"].append(fid)
                for it in self.wm.items:
                    if it["text"] == item["text"] and it.get("fact_id") is None:
                        it["fact_id"] = fid
                self._maybe_calibrate(item["t"])
            rec["relation_facts"].extend(
                f for f in out["facts"] if f["key"].split("|", 1)[1]
                .startswith("关系."))
            rec["relations_retracted"].extend(out.get("relations_retracted") or [])
        self.log.append(rec)
        return rec

    def _appraise(self, valence, novelty):
        apr = Appraisal(desirability=valence, likelihood=1.0,
                        agency="other" if valence < 0 else "circumstance",
                        novelty=novelty, goal_relevance=0.8)
        delta = appraise_to_pad(apr)
        if self.affect.buf.offer(self.affect.t_ms + 1.0, delta):
            # 速行窗口接受了的冲击，随本 tick 的 OU 步进注入状态
            self._drive = delta if self._drive is None else self._drive + delta

    def _maybe_calibrate(self, t):
        self.writer.maybe_calibrate(self.schedule, t)

    # ------------------------------------------------ 便利
    def set_channel_S(self, S, note="manual"):
        """逐渠道标定后冻结 [SUMMARY §6.4]。"""
        self.store.calib_set("session", "kernel", "S", S, note=note,
                             fitted_t=self.t)
        self.retriever.S = S
        self.schedule.freeze()
