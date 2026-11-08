# -*- coding: utf-8 -*-
"""冲突检测器 + 慢系统 [DESIGN §5]。

关键不是有两个系统，是「什么时候切」：System 2 默认不启动，被冲突或意外唤醒。
六个信号 c1-c6，任一超过阈值才切慢系统，否则永远走快系统。

慢系统必须给显式预算（LLM 的深思考不天然昂贵）[DESIGN §5.1 陷阱提醒]。
"""
from . import config

THINK_WORDS = ("想想", "思考", "认真", "仔细", "想清楚")


class ConflictDetector:
    def __init__(self, store):
        self.store = store

    def check(self, question_text=None, answer=None, retrieval=None,
              arousal=0.0, valence=0.0):
        """返回触发的信号名列表。"""
        flags = []
        # c1 检索结果互相矛盾 / c2 置信度低（最高分与次高分接近）
        if retrieval and len(retrieval) >= 2:
            top, second = retrieval[0][1], retrieval[1][1]
            denom = max(abs(top), 1e-9)
            if abs(top - second) / denom < config.CONFLICT_THRESH["c1_tie"]:
                flags.append("c2")
        if answer is not None and getattr(answer, "stale", False):
            flags.append("c1")       # 支撑全失效：检索结果与「现在」矛盾
        # c3 与自我叙事冲突：答案涉及的正是自我叙事里那条、但来源级别低
        if answer is not None and getattr(answer, "rel", None) is not None \
                and answer.rel >= 3 and self.store.self_narrs():
            flags.append("c3")
        # c4 情绪强度过高
        if arousal > config.CONFLICT_THRESH["c4_arousal"] or abs(valence) > 0.8:
            flags.append("c4")
        # c5 这问题我上次答错过
        if question_text and self.store.last_wrong_qa(question_text) == "wrong":
            flags.append("c5")
        # c6 用户明确要求「想想」
        if question_text and any(w in question_text for w in THINK_WORDS):
            flags.append("c6")
        return flags


class SlowSystem:
    """慢系统：有显式 token 预算。做什么：多路复核 + 来源交叉 + 措辞收敛。"""

    def __init__(self, store, retriever):
        self.store = store
        self.retriever = retriever
        self.spent = 0

    def reset(self):
        self.spent = 0

    @property
    def budget_left(self):
        return config.SLOW_BUDGET_TOKENS - self.spent

    def deliberate(self, entity, predicate, question_text=None):
        """复核一条事实：当前值 + 历史一致性。费用记预算。"""
        self.spent += 400
        rows = self.store.history(entity, predicate)
        active = [r for r in rows if r["status"] == "active"]
        if not active:
            return None
        # 交叉验证：看同值是否被多个独立 obs 说过
        vals = {}
        for r in rows:
            vals.setdefault(r["value"], 0)
            vals[r["value"]] += 1
        best = max(vals.items(), key=lambda kv: kv[1])
        hedged = best[0] != active[-1]["value"] or len(active) == 0
        return dict(value=active[-1]["value"], n_history=len(rows),
                    hedged=bool(hedged))
