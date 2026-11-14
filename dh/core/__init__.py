# -*- coding: utf-8 -*-
"""core：心智循环层 [DESIGN §1]。

七步 tick（tick.py）+ 它的直接依赖：工作记忆（working_mem.py）、
冲突检测与慢系统（conflict.py）、第 7 步离线巩固/心智游移（offline.py）。

情绪不在本层：状态机（emotion.py）、调制面（disposition.py）、性格慢塑形
（persona.py）都在 dh/limbic/（右脑）——大脑各节点经 limbic.disposition
拿调制参数，不直读 PAD。此处 emotion.py 只是兼容 shim。
"""
