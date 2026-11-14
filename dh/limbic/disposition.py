# -*- coding: utf-8 -*-
"""情绪 → 行为倾向：把 PAD 从一个「形容词」变成「执行参数」。

为什么需要这一层（实测出来的，不是设计洁癖）：
    改之前，情绪只影响输出端——语音 rate/pitch/energy、tone 枚举、
    提示词里那一行「有点低落」。同一句话在 PAD(-0.9,-0.9,-0.9) 与
    PAD(0.9,0.9,0.9) 下跑一轮，走哪条腿、调几次 LLM、经过哪些节点
    完全相同；检索的 mood 参数在 dh/context.py:99 与 dh/llm_bridge.py:42
    都硬编码 0.0，hybrid_score 里 0.2 权重的情绪路永远贡献 0；
    工作记忆的 salience 把 arousal 当常数加，排序不变。
    → 那时情绪是「怎么说」的着色器，不是「做什么」的控制器。

这一层做的事：每个行为面给一个数，数字进的是**控制流**，不是措辞。

    面             来源        落到哪
    mood          P           检索的情绪一致偏移（recall(mood=)）
    focus         A           注意力里 |valence| 的放大指数（挤掉谁）
    talkativeness A,P         句子预算 / 要不要展开
    directness    D           断言 vs 先确认
    patience      P,A         听不懂时追问还是敷衍、要不要跳过寒暄
    breadth       A           这次检索开多宽的筐（k_gate 缩放）
    gate          A(+惊讶)    这次经历记得多深（情节编码强度 obs.salience；
                              事实照旧全写、经历行照旧留底，见 outlet_memory_gate）

设计纪律：所有映射都是**单调、有界、可在极值处查表验证**的初等函数，
不用学习参数——学出来的映射没法在代码里 review，也没法解释「为什么
这会儿它话少」。
"""
import math

import numpy as np

from .. import config


def _clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


class Disposition:
    """某一刻的行为倾向快照。不可变语义：每次现算，不缓存（2 µs）。"""

    __slots__ = ("p", "a", "d", "mood", "focus", "talkativeness",
                 "directness", "patience", "breadth", "gate", "mode",
                 "label")

    def __init__(self, pad, surprise=0.0, attention=1.0):
        self.p, self.a, self.d = float(pad.p), float(pad.a), float(pad.d)
        # 情绪一致检索：正性情绪偏想好事，负性偏想糟心事。
        # 直接用 P 而不是它的符号——「有点高兴」和「非常高兴」想起的东西
        # 浓度不同，这是连续量。
        self.mood = self.p
        # 注意力锐化：唤醒把情绪显著性拉开差距（见 working_mem.salience）。
        # 0.4 是「完全平静也还剩一点区分度」，不让排序退化成纯 |valence|。
        self.focus = _clip(0.4 + 0.6 * self.a)
        # 想说多少：唤醒给量，愉悦再往上加一点（高兴时话多）。
        self.talkativeness = _clip(0.5 + 0.45 * self.a + 0.25 * self.p)
        # 直接程度：D 高就敢下断言，D 低就先确认再说话。
        self.directness = _clip(0.5 + 0.5 * self.d)
        # 耐心：心情差 = 没耐心，亢奋 = 坐不住。p 的系数必须是**正**的：
        # 旧写法 `-0.45*p` 在 p=-0.9 时算出 +0.405，等于「心情越差越有耐性」，
        # 实测烦躁态的 patience 高达 0.675。
        self.patience = _clip(0.55 + 0.30 * self.p - 0.45 * self.a)
        # 检索宽度：平静时收窄（够用就行），激动时放宽（联想多）。
        # 放在 [0.6, 1.5] × k_gate，两端都实测过不会把主路挤爆。
        self.breadth = _clip(1.0 + 0.45 * self.a, 0.6, 1.5)
        # 记忆门：只给情节记忆用（事实记忆一律全写，见 writer 注释）。
        self.gate = float(np.clip(
            config.EMO_GATE_W_A * _clip(self.a)
            + config.EMO_GATE_W_S * _clip(surprise)
            + config.EMO_GATE_W_N * _clip(attention), 0.0, 1.0))
        self.mode = self._mode()
        self.label = self._label()

    def _mode(self):
        """话语模式。这是控制流，不只是措辞：
        terse     —— 句子预算压到最短，不展开、不举例
        normal    —— 正常
        elaborate —— 愿意多说、愿意联想
        defer     —— 敷衍：给个最短回应，不追问、不调大模型

        `defer` 的旧判据是 `patience < 0.3 and talkativeness < 0.45`，而这两个
        条件在数学上互斥（前者要求 p、a 为正得够多，后者要求为负得够多），
        八个状态实测一次都没进过——敷衍分支是死代码。现在改按「说不说得动」
        单独判：低唤醒（困）或持续低落时，话语量本来就会塌到 0.3 以下。
        """
        if self.talkativeness < 0.30:
            return "defer"
        if self.talkativeness < 0.45 or self.p < -0.55:
            return "terse"
        if self.talkativeness > 0.72 and self.p > 0.25:
            return "elaborate"
        return "normal"

    def _label(self):
        """一个词概括此刻状态（给日志和提示词用）。

        顺序有讲究：旧写法把 `a < 0.15` 放在最后当兜底，而 PAD(0,0,0) 的 a
        就是 0，于是「最初始的平静」被标成了「发困」。发困必须是「低唤醒
        且心情不差」，而且要排在低落之前判。
        """
        if self.p > 0.45 and self.a > 0.5:
            return "兴奋"
        if self.p > 0.3:
            return "不错"
        if self.p < -0.5 and self.a > 0.4:
            return "烦躁"
        if self.p < -0.4 and self.d > 0.4:
            return "心里有数"
        if self.p < -0.4:
            return "低落"
        if self.a < -0.35:
            return "发困"
        return "平静"

    # ---------------------------------------------------------- 消费面
    def scaled_k(self, k):
        """按情绪宽度缩放检索条数。最终值至少 4、最多 2k，避免极端值把
        候选池打到没有或撑爆（实测 k=8 时宽度 >2 倍会把 fact 全挤出）。"""
        n = int(round(k * self.breadth))
        return max(4, min(n, 2 * k))

    def salience(self, valence, base=0.5):
        """工作记忆的显著性：唤醒放大情绪极端度，不再是同加常数。

        旧式 base+|v|+a 里 a 对每条候选都加同一个数，**排序与 a 无关**，
        所以「情绪决定挤掉谁」这句话在旧代码里是假的。改成
        base + |v| * (0.5 + focus)：唤醒高时 |v| 大的那条被拉得更开，
        唤醒低时所有候选挤在一起（谁被挤掉近乎随机，符合「发困时记不住
        重点」）。
        """
        return float(base + abs(float(valence)) * (0.5 + self.focus))

    def prompt_line(self):
        """给 LLM 的行为约束（不是形容词）。每条都是祈使句，可检验。"""
        bits = []
        if self.mode == "terse":
            bits.append("这句话别超过十个字，不要展开也不要举例")
        elif self.mode == "elaborate":
            bits.append("可以多说两句，愿意联想和追问")
        elif self.mode == "defer":
            bits.append("简短应一声就行，别追问也别展开")
        if self.directness < 0.35:
            bits.append("拿不准的要先说「我记得」，不要当成确定的事")
        elif self.directness > 0.7:
            bits.append("可以直接说，不用加「我记得」这类铺垫")
        if self.patience < 0.3:
            bits.append("别寒暄、别反问，直接给结论")
        if not bits:
            return ""
        return "【此刻的分寸】" + "；".join(bits) + "。"


def disposition(dh, surprise=0.0, attention=1.0):
    """从当前情绪算倾向。2 µs 量级，不用缓存。"""
    return Disposition(dh.affect.state, surprise=surprise, attention=attention)


def gate(pad, surprise=0.0, attention=1.0):
    """只要记忆门时的快捷方式（毫秒级调用点用）。"""
    return Disposition(pad, surprise, attention).gate
