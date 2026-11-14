# -*- coding: utf-8 -*-
"""dh.brain：主图（大脑）——记忆存取控制整个 agent 的落点。

三模块分工（不写一个文件到底）：
    policy.py   路由策略    分支条件全部读记忆/心智状态，纯函数、可单测
    nodes.py    节点实现    guard→perceive→recall→plan→compose→affect→finish→speak
    graph.py    图与执行器  节点表/锁边界/epoch 协议/begin-run 拆分；describe() 可视

对外入口与旧 dh/turn.py 完全一致（run/begin）；dh/turn.py 现在只是
兼容 shim。web/chat_service 的慢路径尚未并入本图（它带 langgraph 语音
工具循环与自己的 epoch 记账），但它的决策闸门已改用本包 policy——
大脑先统一，步子后并。
"""
from .graph import begin, run, describe
from . import policy

__all__ = ["begin", "run", "describe", "policy"]
