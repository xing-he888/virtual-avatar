# -*- coding: utf-8 -*-
"""dh：数字人记忆/情绪系统的实现包。

对应关系（研究文档 -> 模块）：
    DIGITAL_HUMAN_DESIGN.md   §1 七步循环      -> tick.py
                              §2 五个存储区    -> store.py / working_mem.py
                              §3 检索          -> retrieve.py / kernel.py / calib.py
                              §4 情绪          -> emotion.py
                              §5 思维          -> conflict.py / offline.py
                              §6 幻觉治理      -> retrieve.py(措辞/降级) / store.py(rel)
    DIGITAL_HUMAN_STORAGE.md  存储层           -> store.py / writer.py
    DIGITAL_HUMAN_MEMORY_GRAPH.md 激活层       -> graph_layer.py
    PROJECT_SUMMARY.md        全部工程参数      -> config.py（每个数带出处）
"""
from .store import MemoryStore
from .tick import DigitalHuman

__all__ = ["MemoryStore", "DigitalHuman"]
__version__ = "0.1.0"
