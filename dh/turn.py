# -*- coding: utf-8 -*-
"""兼容 shim：一轮对话的节点引擎已搬到 dh/brain/（policy/nodes/graph 三模块）。

搬迁理由：主图是「记忆存取控制整个 agent」的大脑，原来一个文件里
节点实现、路由 if 链、执行器三层耦死；现在路由是可单测的纯函数
（brain/policy.py），图是可打印的数据（brain/graph.py describe()）。
本文件只为既有导入（main.py 的 begin/run、web 的 _trust、测试的
node_* 名字）保持原样可用——新代码请直接 import dh.brain。
"""
from .brain.graph import begin, run, describe          # noqa: F401
from .brain.nodes import (                             # noqa: F401
    node_guard, node_perceive, node_recall, node_plan, node_compose,
    node_affect, node_finish, node_speak,
    _ensure_heard, _trust,
)
from .brain.policy import is_filler, decide            # noqa: F401

__all__ = ["begin", "run", "describe", "is_filler", "decide",
           "node_guard", "node_perceive", "node_recall", "node_plan",
           "node_compose", "node_affect", "node_finish", "node_speak",
           "_ensure_heard", "_trust"]
