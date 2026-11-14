# -*- coding: utf-8 -*-
"""工作记忆：4 个槽，硬上限 [DESIGN §2 规则 1]。

不是「建议 4 个」，是第 5 个进来必须挤掉一个，挤掉谁由情绪显著性决定。
不许开后门（DESIGN §8 第 6 条）——开后门瓶颈消失，拟人机制跟着消失。
"""
from .. import config


class WorkingMemory:
    def __init__(self, slots=None):
        self.slots = config.WM_SLOTS if slots is None else slots
        self.items = []          # 按 tts(进入时刻) 排列；每项 dict(obs_id,text,salience,t)

    def push(self, item):
        """item: dict(obs_id, text, salience, t)。满了挤掉显著性最低的。"""
        evicted = None
        if len(self.items) >= self.slots:
            worst = min(range(len(self.items)), key=lambda i: self.items[i]["salience"])
            if self.items[worst]["salience"] >= item.get("salience", 0.0):
                return item, False       # 新来的不如旧的重要，新来的自己被挤
            evicted = self.items.pop(worst)
        self.items.append(item)
        self.items.sort(key=lambda x: x["t"])
        assert len(self.items) <= self.slots, "工作记忆后门被打开"
        return evicted, True

    def salience(self, valence, arousal=0.0, base=0.5, focus=None):
        """情绪显著性：|效价| × 唤醒锐化 + 底数。挤掉谁由它决定。

        ⚠ 旧写法 `base + |v| + max(a,0)` 里 arousal **对每条候选都加同一个
        常数**，排序与它无关——所以「唤醒决定挤掉谁」在旧代码里是假的
        （实测 arousal=0 与 0.9 的差值是同一个 0.70）。

        现在 arousal 通过 focus 作用在 |valence| 上：
            base + |v| · (0.5 + focus)
        focus 由 dh/limbic/disposition.py 按 0.4+0.6·A 给出，于是唤醒高时
        |v| 大的那条被拉得更开（记住戳心的那个），唤醒低时所有候选挤在
        一起（发困时记不住重点，挤掉谁近乎随机）。focus=None 时退化为
        旧公式，供老测试与实验脚本使用。
        """
        if focus is None:
            return base + abs(valence) + max(arousal, 0.0)
        return float(base + abs(float(valence)) * (0.5 + float(focus)))

    def texts(self):
        return [it["text"] for it in self.items]
