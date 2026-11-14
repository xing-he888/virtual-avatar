# -*- coding: utf-8 -*-
"""extract：抽取层——一句话 -> (来源, 主体, 谓词, 值)。

    rules.py  Tier 0：正则 + 别名表，零成本、确定性
    llm.py    Tier 1：LLM 抽取，五道闸门，只处理正则吃不下的话

LLM 发现的新实体回写别名词典 -> 下次正则快路直接命中（自学习）。
"""
from .rules import (
    Extractor, detect_relations, has_kinship_term, is_segment_boundary,
    normalize_value, REL_CONFLICT, self_intro_name,
)

__all__ = ["Extractor", "detect_relations", "has_kinship_term",
           "is_segment_boundary", "normalize_value", "REL_CONFLICT",
           "self_intro_name"]
