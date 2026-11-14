# -*- coding: utf-8 -*-
"""dh.limbic：右脑（边缘系统）——情绪对大脑每一步的调制器。

架构定位（用户约定）：大脑（dh/brain/，主图）负责决策与执行；右脑负责
「这一刻是什么状态、每一步该带着什么劲」。右脑不写库、不说话——它只
产数字；大脑的每个节点按需读面，情绪由此进入**控制流**而不只是措辞。

三个组件：
    emotion.py       状态机：PAD 表征 + OCC/顺序评价 + OU 精确离散化 +
                     四出口（语言读 P+D / 语音读 A / 表情读 A+P / 记忆门读 A）
    disposition.py   调制快照：PAD -> 七个行为面，全部单调有界可查表
    persona.py       慢塑形：经历极性以 0.1 学习率拖动 OU 基线、信任值
                     （真值入 calib 只插不改，persona.md 只是视图）

右脑 -> 大脑的调制接口（大脑只准从这里拿情绪，不直接读 PAD 裸值）：

    面             来源     大脑消费点
    mood          P        recall 的情绪一致检索：谁能进候选池（context.collect_context）
    focus         A        注意显著性：工作记忆挤掉谁（core/tick）
    talkativeness A,P      腿路选择：defer/terse/elaborate（brain/policy.decide）
    directness    D        措辞：垫不垫「我记得」（memory/retrieve.Answer.phrase）
    patience      P,A      敷衍用语轮换（brain/policy.defer_line）、跳过寒暄
                           （prompt_line 经 llm_bridge.tone_block 进提示词）
    breadth       A        检索开多宽：k_gate 缩放（context.collect_context）
    gate          A(+惊讶) 情节编码强度（obs.salience）：记得多深——读出侧
                           「想起更快（EMO_SAL_GAIN）/ 忘得更慢（SAL_AGE_SCALE）/
                           塑形更重（persona.shape 加权）/ 游移偏爱（offline 种子）」。
                           事实记忆照旧一律全写（纪律 3），obs 永远留底，门控
                           不删行不跳写

情绪的进入口（感知）：Tier 0 小词典（离线兜底）+ LLM <appraisal> 协议
（主路：模型按性格底稿指认评价维度，appraise_to_pad + OU 负责结论——
右脑只让 LLM 指认，不让它发明情绪）。
"""
from .emotion import (                                # noqa: F401
    PAD, PAD_DIM, Appraisal, appraise_to_pad,
    OUDynamics, ReactionBuffer, AffectSystem,
    outlet_language, outlet_speech, outlet_face, outlet_memory_gate,
    autocov, estimate_theta_c2c1, estimate_theta_raw,
)
from .disposition import Disposition, disposition, gate   # noqa: F401
from .persona import Persona, clamp                       # noqa: F401

__all__ = [
    "PAD", "PAD_DIM", "Appraisal", "appraise_to_pad",
    "OUDynamics", "ReactionBuffer", "AffectSystem",
    "outlet_language", "outlet_speech", "outlet_face", "outlet_memory_gate",
    "autocov", "estimate_theta_c2c1", "estimate_theta_raw",
    "Disposition", "disposition", "gate", "Persona", "clamp",
]
