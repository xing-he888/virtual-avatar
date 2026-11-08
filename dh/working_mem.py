# -*- coding: utf-8 -*-
"""工作记忆：4 个槽，硬上限 [DESIGN §2 规则 1]。

不是「建议 4 个」，是第 5 个进来必须挤掉一个，挤掉谁由情绪显著性决定。
不许开后门（DESIGN §8 第 6 条）——开后门瓶颈消失，拟人机制跟着消失。
"""
from . import config


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

    def salience(self, valence, arousal, base=0.5):
        """情绪显著性：|效价| + 唤醒 + 底数。挤掉谁由它决定。"""
        return base + abs(valence) + max(arousal, 0.0)

    def texts(self):
        return [it["text"] for it in self.items]
