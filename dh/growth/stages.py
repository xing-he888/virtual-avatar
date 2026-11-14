# -*- coding: utf-8 -*-
"""发展阶段：像人一样从出生慢慢长大，而不是 XP 升级。

四个阶段（初生 → 幼生 → 成长 → 成熟），解锁条件 = 累计交互轮数 + 日历
天数 + 经历多样性（聊过多少个不同的实体）。三条铁律：

    1. 只进不退（持久化在 calib，重启/清屏都不倒退）——Replika 2023 事件
       的教训：人格/关系的连续性是命，倒退 = 毁掉用户感知的成长
    2. 不做 XP 游戏化：阶段是发展状态不是奖励（学术评估点名游戏化制造
       依附）；用户感知到的应该是「他长大了」而不是「他升级了」
    3. 阶段调制成长本身：幼生期学习率最高（可塑窗口），成熟期趋零（定性）

每阶段限制的行为面：
    初生   话短、游移不启动、信任建立快、学习率 1.5（天生学得快）
    幼生   句子变长、游移开启（开始自己联想）、学习率 2.0（可塑窗口峰值）
    成长   联想宽、记忆门更挑、学习率 1.0（性格开始定型）
    成熟   特质近乎冻结（学习率 0.2）、只剩极慢漂移
"""
from .. import config

STAGES = ("初生", "幼生", "成长", "成熟")

# 解锁条件（全部满足才进下一阶段）。阈值可调——这里都是工程默认，
# 没有真实数据背书，出处标注为「设计值」。
STAGE_RULES = {
    "幼生": dict(turns=200, days=3, topics=3),
    "成长": dict(turns=1000, days=14, topics=5),
    "成熟": dict(turns=5000, days=60, topics=8),
}

STAGE_LIMITS = {
    "初生": dict(learn=1.5, wander=False, talk_bias=-0.18, trust_mult=1.5,
                 line="你现在还在初生期：说话要短、只聊眼前的事，别联想别跑题"),
    "幼生": dict(learn=2.0, wander=True, talk_bias=-0.05, trust_mult=1.2,
                 line="你在幼生期：可以多说两句、开始自己联想，但别装老成"),
    "成长": dict(learn=1.0, wander=True, talk_bias=0.0, trust_mult=1.0,
                 line="你在成长期：联想和好奇都可以放开，性格开始定型"),
    "成熟": dict(learn=0.2, wander=True, talk_bias=0.0, trust_mult=1.0,
                 line="你在成熟期：性格基本定性，只随经历极慢地变"),
}


class Stage:
    """某只小星的发展阶段：算（从经历统计）、存（只进不退）、给限制。"""

    def __init__(self, store):
        self.store = store

    # ------------------------------------------------ 统计
    def _stats(self):
        turns = self.store.con.execute(
            "SELECT COUNT(*) FROM obs WHERE kind='told' AND source='用户'"
        ).fetchone()[0]
        first = self.store.con.execute(
            "SELECT MIN(t) FROM obs").fetchone()[0] or 0.0
        last = self.store.con.execute(
            "SELECT MAX(t) FROM obs").fetchone()[0] or 0.0
        days = max((last - first) / 86400.0, 0.0)
        topics = self.store.con.execute(
            "SELECT COUNT(DISTINCT entity) FROM fact WHERE entity IS NOT NULL"
        ).fetchone()[0]
        return dict(turns=int(turns), days=round(days, 2), topics=int(topics))

    def _earned(self):
        """按条件算「配得上」的阶段下标。"""
        s = self._stats()
        idx = 0
        for i, name in enumerate(STAGES[1:], start=1):
            r = STAGE_RULES[name]
            if (s["turns"] >= r["turns"] and s["days"] >= r["days"]
                    and s["topics"] >= r["topics"]):
                idx = i
        return idx, s

    # ------------------------------------------------ 读（只进不退）
    def current(self):
        """当前阶段 dict(name, idx, stats, limits, line)。持久化，只进不退。"""
        earned, stats = self._earned()
        row = self.store.calib_latest("growth", "stage", "idx")
        idx = max(earned, int(row) if row is not None else 0)
        if row is None or idx > int(row):
            self.store.calib_set("growth", "stage", "idx", idx,
                                 note="stage", fitted_t=0.0)
        name = STAGES[idx]
        lim = dict(STAGE_LIMITS[name])
        lim["stats"] = stats
        lim["earned_next"] = self._next_gap(idx, stats)
        return dict(name=name, idx=idx, **lim)

    @staticmethod
    def _next_gap(idx, stats):
        """距下一阶段还差什么（给前端/自述用；成熟返回 None）。"""
        if idx >= len(STAGES) - 1:
            return None
        r = STAGE_RULES[STAGES[idx + 1]]
        return dict(turns=max(r["turns"] - stats["turns"], 0),
                    days=round(max(r["days"] - stats["days"], 0.0), 1),
                    topics=max(r["topics"] - stats["topics"], 0))
