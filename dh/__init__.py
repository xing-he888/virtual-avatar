# -*- coding: utf-8 -*-
"""dh：数字人记忆/情绪系统的实现包。

分层（详见 ARCHITECTURE.md）：
    core/     心智循环层  七步 tick、工作记忆、冲突/慢系统、离线巩固
    limbic/   右脑        情绪状态机（emotion）+ 调制面（disposition）+ 性格慢塑形（persona）
    memory/   记忆层      存储、写回、检索、时间核、激活图、标定、值等价裁决
    extract/  抽取层      正则规则（Tier 0）与 LLM 补记（Tier 1）
    brain/    主图（大脑）policy 路由策略 / nodes 节点 / graph 执行器
    config.py 全部工程参数（每个数带出处）
    world.py  合成对话世界（评测协议）
    context.py            检索上下文的唯一收集点（一次会话只召回一次）
    turn.py               兼容 shim（begin/run/_trust 原样指向 dh.brain）
    llm_bridge.py         记忆系统 -> LLM 的桥（提示词/韵律/后台补记）

研究文档 -> 模块对应：
    DIGITAL_HUMAN_DESIGN.md   §1 七步循环      -> core/tick.py
                              §2 五个存储区    -> memory/store.py / core/working_mem.py
                              §3 检索          -> memory/retrieve.py / kernel.py / calib.py
                              §4 情绪          -> limbic/emotion.py（状态机）
                                                 limbic/disposition.py（调制面）
                                                 limbic/persona.py（慢塑形）
                              §5 思维          -> core/conflict.py / offline.py
                              §6 幻觉治理      -> memory/retrieve.py(措辞/降级) / memory/store.py(rel)
    DIGITAL_HUMAN_STORAGE.md  存储层           -> memory/store.py / memory/writer.py
    DIGITAL_HUMAN_MEMORY_GRAPH.md 激活层       -> memory/graph_layer.py
    PROJECT_SUMMARY.md        全部工程参数      -> config.py（每个数带出处）
"""
from . import config                    # 先于子包：子包里全是 from .. import config
from .memory.store import MemoryStore
from .core.tick import DigitalHuman

__all__ = ["MemoryStore", "DigitalHuman"]
__version__ = "0.2.0"
