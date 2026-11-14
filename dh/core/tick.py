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
import json
import time

import numpy as np

from .. import config
from ..memory.calib import CalibSchedule
from .conflict import ConflictDetector, SlowSystem
from ..limbic.disposition import disposition, gate as emo_gate
from ..limbic.emotion import AffectSystem, Appraisal, PAD, appraise_to_pad
from ..extract import Extractor
from ..memory.graph_layer import ActivationGraph
from ..memory.qroute import QuestionRouter
from .offline import Offline
from ..memory.retrieve import Answer, Retriever
from ..memory.store import MemoryStore
from ..memory.writer import Writer

QUESTION_WORDS = ("吗", "呢", "？", "?", "多少", "什么", "几", "怎么", "多大")
# 「多大」：实测「我今年多大了」在旧表一个词都中不了（多少/几/怎么全不沾），
# 问句路由和 is_question 都会漏——补上。误伤面极小（陈述句里带「多大」的
# 本来也多半在问尺寸/年纪，跳过写回是正确行为）。

# 情绪的墙钟同步上限（分钟）：隔了三天没聊，也按「过了一天」封顶——
# OU 精确离散化下 dt=1440 与 dt=2880 的落点都是性格基线，封顶只是省事
_AFFECT_SYNC_MAX_MIN = 1440.0


def looks_like_question(text):
    return any(w in text for w in QUESTION_WORDS)


class DigitalHuman:
    """事件驱动的七步循环：hear()/ask() 触发一轮 tick，idle() 触发第 7 步。"""

    def __init__(self, store=None, alias=None, embed_fn=None, seed=0,
                 S=None, sigma=None, debug=False, valence_fn=None):
        from ..memory import embed as embed_mod
        if embed_fn is None:
            # 有本地 bge 就用 bge（真语义），没有就退回 hash（字面重叠）。
            # 模型目录：<repo>/models/bge-small-zh-v1.5
            embed_fn, _ = embed_mod.make_embed_fn()
        # 效价评价默认小词典；真实评价器/LLM 打分从这里注入
        self.valence_fn = valence_fn or Extractor.valence
        self.store = store or MemoryStore()
        self.writer = Writer(self.store, alias=alias, embed_fn=embed_fn)
        self.graph = ActivationGraph(self.store)
        # 检索层要知道图：语义召回拿到种子后要能在图上走两跳，
        # 才不会漏掉"同一个事件/同一个人"但措辞完全不像的记忆
        self.retriever = Retriever(self.store, embed_fn=embed_fn, S=S, sigma=sigma,
                                   graph=self.graph)
        # 问句路由器：L0 属主消解/线索子串 + L1 语义路由；L2 的 LLM 兜底
        # 在 llm_bridge（dh 不持 LLM）。种子线索幂等，启动补种一次
        self.qrouter = QuestionRouter(self.store, embed_fn,
                                      extractor=self.writer.ex)
        self.qrouter.seed()
        self.store.commit()
        self.affect = AffectSystem()
        self.affect.rng = np.random.default_rng(seed + 1)   # 情绪噪声也进种子
        # 出厂≠定死：库里已有被经历塑形过的情绪基线，开机即恢复（右脑慢塑形）
        from ..limbic.persona import Persona
        self.affect.dyn.mu = np.array(Persona(self).mu())
        # 情绪跨会话持久：状态 + 墙钟锚存 meta——重启后第一条消息进来，
        # 先按「离开时的真实时间差」让情绪演化到位（隔夜火气自然消）
        self._affect_wall = time.time()
        raw = self.store.get_meta("affect_state")
        if raw:
            try:
                st = json.loads(raw)
                self.affect.state = PAD(float(st["p"]), float(st["a"]),
                                        float(st["d"])).clipped()
                self._affect_wall = float(st["wall"])
            except (ValueError, TypeError, KeyError):
                pass
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

    # ------------------------------------------------ 情绪的墙钟同步
    def _sync_affect_wall(self, now):
        """把情绪按「距上次的真实时间差」演化到位。

        一律交给 `affect.decay()`：短间隔走随机 OU 步（人就在旁边，情绪该
        有自然抖动），长间隔走确定性均值回归（一次 exp，精确、不抽签）。

        **不要试图用切小步来「让衰减真的发生」**——实测过，切成 60 秒步对
        长间隔毫无改善：10 分钟时精确值 A=+0.156，一步跨 +0.113/sd 0.473，
        切 10 步 +0.079/sd 0.469。噪声按 sqrt(n) 累积，总量和一步跨同阶。
        「第二天就忘」本来就是方程的解，不需要靠抽签实现。
        """
        dt_ms = min(max((now - self._affect_wall) * 1000.0, 0.0),
                    _AFFECT_SYNC_MAX_MIN * 60000.0)
        if dt_ms > 1000.0:
            self.affect.decay(dt_ms)
        self._affect_wall = max(self._affect_wall, now)

    def affect_clock(self, dt_ms=None):
        """常驻情绪时钟：没人说话时也让情绪继续演化（2 µs/次）。

        为什么必须有：`tick()` 里 `affect.tick()` 在 `for item in attended:`
        循环内，没有感知进注意就一步都不走。实测 dh.tick(1.0) 与
        dh.tick(60.0) 推进的情绪时间都是 0 ms——离开一小时再回来，情绪是在
        开口那一刻才跳一次，中间冻住。拟人从这里就断了。

        返回推进后的 PAD（调用方可以拿去推 SSE）。
        """
        dt_ms = config.EMO_CLOCK_MS if dt_ms is None else float(dt_ms)
        self.affect.decay(dt_ms)
        self._affect_wall = max(self._affect_wall, time.time())
        return self.affect.state

    def save_affect(self):
        """情绪状态 + 墙钟锚写入 meta（跨重启持久，配合墙钟同步）。"""
        self.store.set_meta("affect_state", json.dumps(
            dict(p=self.affect.state.p, a=self.affect.state.a,
                 d=self.affect.state.d, wall=time.time())))

    # ------------------------------------------------ 对外入口
    def hear(self, text, speaker="用户", t=None, valence=None):
        """有人说了句话。跑一轮完整 tick（第 5 步多半是「什么都不做」）。"""
        t = time.time() if t is None else float(t)
        self.t = max(self.t, t)
        # 时间先走：距上次的真实时间差里，情绪已经演化（隔夜回基线）
        self._sync_affect_wall(t)
        if valence is None:
            valence = self.valence_fn(text)
        self.sensory.append(dict(text=text, speaker=speaker, t=t, valence=valence,
                                 is_question=looks_like_question(text)))
        return self.tick()

    def ask(self, text, t=None, record=True):
        """问它一句话：检索 -> 评估 -> 反应。返回 (Answer, 措辞, 诊断)。

        record=False：调用方已经把这句问话写进经历层了（web 的
        ChatServer._ensure_heard 在 hear 之后补的那条），别再写第二遍。
        否则同一句话会在同一秒落两条一模一样的 obs——问句被 tick 跳过
        写回，ask 又自己写一条，重复就此产生。
        """
        t = self.t if t is None else float(t)
        self.t = max(self.t, t)
        rec = dict(t=t, kind="ask", text=text)
        # 3 检索：问题 -> (实体, 谓词) -> 点查（微秒级 [STORE §5.2]）
        # 路由器 = 原小抄 + 属主消解 + 自学问法线索（语义）——问句侧的开放
        # 词表不再靠枚举；route 失败仍回落 parse（陈述式问答）
        parsed = self.qrouter.route(text) or self.ex.parse(text)
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
        if record:
            self.store.add_obs(t, "told", "text", text, source="用户",
                               rel=1, seg_bnd=0.0)
        outcome = "abstain" if ans.abstain else ("degraded" if ans.degraded else "ok")
        self.store.log_qa(text, t, outcome)
        # 措辞要读此刻的把握感：D 低就垫一句「我记得」。
        # 之前这里写死 ans.phrase()，于是「没底气」这件事在快路径上不存在。
        phrased = ans.phrase(directness=disposition(self).directness)
        rec["answer"] = phrased
        self.log.append(rec)
        return ans, phrased, diag

    def idle(self, t=None, rng=None):
        """第 7 步：巩固 + 心智游移。"""
        t = self.t if t is None else float(t)
        self._sync_affect_wall(time.time())   # 情绪永远按真实墙钟衰减
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
        #    显著性由「此刻的倾向」算：唤醒高时 |valence| 被放大（记住戳心的
        #    那个），唤醒低时所有候选挤在一起（发困时记不住重点）。
        disp = disposition(self)
        batch = alive[:config.ATTEND_PER_TICK]
        attended = []
        for x in batch:
            sal = disp.salience(x.get("valence") or 0.0)
            attended.append(dict(obs_id=None, text=x["text"], salience=sal,
                                 t=x["t"], speaker=x["speaker"],
                                 valence=x.get("valence"),
                                 is_question=x["is_question"]))
        self.sensory = alive[len(batch):]
        rec["attended"] = len(attended)
        # 3 检索 + 4 评估（逐条进工作记忆；4 槽硬上限，挤掉显著性最低的）
        for item in attended:
            self.wm.push(item)
            # 问句不算「意外事件」：它的 novelty 必须为 0，否则每问一次
            # 都在抬高 arousal（appraise_to_pad 里 a = 0.55*novelty + ...），
            # 实测连问 10 句 A 会爬到 +0.12 并且下不来——人问问题不会越问
            # 越亢奋。陈述才带 novelty（新信息才意外）。
            _nov = 0.0 if item.get("is_question") else 0.1
            self._appraise(valence=item.get("valence") or 0.0, novelty=_nov)
            self.affect.tick(dt_s * 1000.0, drive=self._drive)  # 收毫秒
            self._drive = None
        # 5 反应：陈述不触发说话（说话由 ask()/上层驱动）；这里只记状态
        rec["mood"] = dict(p=self.affect.state.p, a=self.affect.state.a,
                           d=self.affect.state.d)
        # 6 写回：把经历写进记忆（事实记忆一律全写，不按唤醒度挑 [DESIGN §4.3 补注]）。
        # 情绪在这里第一次影响「记得多深」：gate(A,惊讶,注意) 落到 obs.salience——
        # 不删行、不跳写，只定编码强度（读出侧：想起更快 / 忘得更慢 / 塑形更重）
        rec["written"] = []
        rec["obs_ids"] = []
        rec["relation_facts"] = []          # 本轮写的关系明细（动态修正确认用）
        rec["relations_retracted"] = []     # 本轮被撤回的冲突旧关系
        for item in attended:
            if item.get("is_question"):
                continue
            _nov = 0.0 if item.get("is_question") else 0.1   # 问句不意外（同上）
            _attn = min(1.0, max(0.0, item.get("salience") or 0.0))
            _emo_sal = emo_gate(self.affect.state, surprise=_nov, attention=_attn)
            out = self.writer.hear(item["text"], item["t"], item["speaker"],
                                   valence=item.get("valence"),
                                   salience=_emo_sal)
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

    def _appraise(self, valence, novelty, agency=None, controllability=None):
        """把一次经历评成 PAD 增量。

        D 维（支配）旧写法是 `agency="other" if valence<0 else "circumstance"`
        且从不给 controllability，于是 D 只有两个取值（负面 -0.280、正面
        0.000）——「笃定程度」实际上只等于「刚才是不是坏事」。这里显式区分
        责任：外部施压（other，D 降）与自己可控（self，D 升）都可能是坏事，
        但一个让人没底、一个让人心里有数。
        """
        # valence=0 且不意外 = 「什么都没发生」，不该往情绪里注入任何东西。
        #
        # 这里踩过坑：appraise_to_pad 把 novelty 单独算进 arousal
        # （a = 0.55*novelty + ...），于是 valence=0.0 + novelty=0.1 也返回
        # (0, +0.044, +0.148)。而 ask() 每问一次就调一次 _appraise(0.0, 0.1)，
        # 本意是「被问到别人的事不算情绪事件」，实际却在稳定抬高 arousal；
        # node_perceive 传 valence=0.0 想让 LLM 单独评价，也因为这一路
        # 先推了一次，等于同一件事计两次（实测 LLM 路径 after=(−0.168,
        # +0.169, +0.006)，那 +0.169 就是这里的白推）。
        if abs(valence) < 1e-9 and novelty < 0.35:
            return
        if agency is None:
            agency = "other" if valence < 0 else "circumstance"
        if controllability is None:
            controllability = 0.35 if valence < 0 else 0.6
        apr = Appraisal(desirability=valence, likelihood=1.0,
                        agency=agency,
                        novelty=novelty, goal_relevance=0.8,
                        controllability=controllability, certainty=0.6)
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
