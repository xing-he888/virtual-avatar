# -*- coding: utf-8 -*-
"""LLM 抽取层：大模型从自然语言里抽出「主体-谓词-值」，但精确性不靠模型自觉。

设计：五道闸门，逐条过，任何一道不过就丢弃（宁可漏记，不可编造）——

    G1 schema 约束    只接受固定 JSON 结构，解析失败整条丢弃（修复一次）
    G2 时态门槛       过去时/传闻/假设不写成事实（obs 里永远有原文）
    G3 值落地锚定     value 必须是原文连续子串——LLM 只能「指认」不能「发明」；
                      偏移错了就按子串搜索修复，找不到原文就拒绝
    G4 谓词白名单     predicate 必须在注册表里，或 "custom"+合法 new_predicate
                      （2-6 字，进注册表）——防止幻觉谓词污染事实表
    G5 置信门槛       conf < 0.6 丢弃

加上存储层的伤害控制：即使一条错误抽取混进来，它也带 src_obs 可溯源、
带 conf 可降权、被下一条断言自然取代（supersede）——不会悄悄破坏真值。

与正则快路的关系：正则（extract.py）是 Tier 0，零成本、确定性；
LLM 是 Tier 1，只处理正则吃不下的话（开放谓词、新实体、复杂句式）。
LLM 发现的新实体回写别名词典 → 下次正则快路直接命中（自学习）。
"""
import json
import re
from dataclasses import dataclass, field

from .rules import Extractor


@dataclass
class Candidate:
    """一条通过闸门的三元组候选。字段与 extract() 的构造一一对应。"""
    subject: str = ""                 # 已规范化的主体（canon 名）
    predicate: str = ""               # 注册表谓词或自定义谓词
    value: str = ""                   # 原文子串（已落地验证）
    start: int = -1                   # 值在原文中的字符偏移（G3 修复后可信）
    end: int = -1
    tense: str = "current"            # current / past / hearsay / hypothetical
    multi: bool = False               # 谓词是否多值（新谓词时由 LLM 建议）
    conf: float = 0.0                 # LLM 自报置信度（已过闸门）
    new_predicate: str = ""           # predicate=="custom" 时的建议谓词名
    is_new_entity: bool = False       # 主体是否新发现的实体
    is_new_predicate: bool = False
    is_relation: bool = False         # 实体-实体关系（值=对方实体，multi 恒真）


def _repair_json(raw):
    """LLM 输出的 JSON 常见病：代码围栏、前后废话。剥掉再试。"""
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no json object")
    return json.loads(m.group(0))


class LLMExtractor:
    """LLM 三元组抽取器。chat_fn(system, user) -> str 由上层注入
    （服务端接 DashScope，测试接 Mock）。"""

    MAX_VALUE_LEN = 30
    MIN_CONF = 0.6

    def __init__(self, chat_fn, registry=None, extractor=None):
        self.chat_fn = chat_fn            # (system, user) -> str
        # 谓词白名单：{pred: {"multi": bool}}；custom 通过后进这里
        self.registry = dict(registry or {})
        self.extractor = extractor        # 正则抽取器（别名自学习用）

    # ------------------------------------------------------------ 提示词
    def build_prompt(self, text, topics):
        preds = "、".join(sorted(self.registry.keys())) or "（暂无，全部走 custom）"
        topic_str = "、".join(topics) if topics else "（无）"
        system = (
            "你是信息抽取器。从对话原文里抽出「主体-谓词-值」三元组，"
            "只抽说话人**明确断言的当前事实**。\n"
            "规则：\n"
            "1. predicate 只能从这些里选：" + preds +
            "。都不合适就填 \"custom\"，并在 new_predicate 里给一个 2-6 字的新谓词。\n"
            "2. value 必须是原文里**逐字出现的连续子串**，并给出字符偏移 start/end。"
            "禁止改写、补全、单位换算、推断出原文没有的值。\n"
            "3. subject 必须是原文出现的实体名或代词；代词按上下文解析"
            "（最近话题实体：" + topic_str + "）。解析出的实体名如果不在原文里，"
            "也照常填（解析代词是允许的推断）。\n"
            "4. 时态：正在说的当前事实填 \"current\"；过去的事填 \"past\"；"
            "听说的传闻（听X说…）填 \"hearsay\"；假设/愿望填 \"hypothetical\"。\n"
            "5. 否定句（不住/不是/没换成…）**不要抽取**——它否认了值但没有断言新值。\n"
            "6. 一句话可以抽多个三元组；抽不出就返回空列表。\n"
            "7. conf 是你对这条抽取的把握 0-1；没把握就低。\n"
            "8. 原文若明确了两个人之间的关系（亲属/朋友/同事/师生…），"
            "在 relations 里输出一条：from_entity、relation（2-6 字）、to_entity。"
            "方向必须是从属称呼：『老王儿子』的父亲是老王 → from=老王儿子, "
            "relation=父亲, to=老王（relation 是 from 对 to 的称呼）。"
            "两个实体名必须出现在原文或上下文实体里；"
            "没有明确关系就不输出 relations。\n"
            '只输出 JSON：{"triples":[{"subject":"", "predicate":"", '
            '"new_predicate":"", "value":"", "start":0, "end":0, '
            '"tense":"current", "multi":false, "conf":0.9}], '
            '"relations":[{"from_entity":"", "relation":"", "to_entity":"", "conf":0.9}]}'
        )
        return system, "对话原文：" + text

    # ------------------------------------------------------------ 闸门
    def _gate_subject(self, c, text):
        """主体：别名表内的直接过；不在表内的必须落地于原文（G3 的主体版）。"""
        canon = self.extractor.canon(c.subject) if self.extractor else None
        if canon:
            c.subject = canon
            return True, None
        if c.subject and c.subject in text and 1 <= len(c.subject) <= 12:
            c.is_new_entity = True
            return True, None
        return False, "subject-not-grounded"

    def _gate_value(self, c, text):
        """G3 值落地：偏移错了就按子串搜索修复；找不到原文整条拒绝。"""
        v = c.value.strip()
        if not v or len(v) > self.MAX_VALUE_LEN:
            return False, "value-empty-or-too-long"
        if 0 <= c.start < c.end <= len(text) and v in text[c.start:c.end]:
            return True, None
        idx = text.find(v)                       # 偏移错 → 搜索修复
        if idx < 0:
            nospace = text.replace(" ", "")
            idx2 = nospace.find(v)
            if idx2 >= 0:                        # 原文带空格的兜底
                c.value = v
                c.start, c.end = idx2, idx2 + len(v)
                return True, None
            return False, "value-not-grounded"
        c.start, c.end = idx, idx + len(v)
        return True, None

    def _gate_predicate(self, c):
        """G4 谓词白名单：注册表内直接过；custom 需要合法 new_predicate。"""
        if c.predicate in self.registry:
            return True, None
        if c.predicate == "custom" and c.new_predicate:
            np_ = c.new_predicate.strip()
            if 2 <= len(np_) <= 6 and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9]+", np_):
                c.predicate = np_
                c.is_new_predicate = True
                self.registry[np_] = {"multi": c.multi}   # 进白名单（自学习）
                return True, None
        return False, "predicate-not-allowed"

    # ------------------------------------------------------------ 主流程
    def extract(self, text, topics=None):
        """返回 (candidates, dropped)；dropped = [(原因, 摘要)] 供调试。"""
        system, user = self.build_prompt(text, topics or [])
        raw = self.chat_fn(system, user)
        dropped = []
        try:
            data = _repair_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return [], [("g1-schema", str(exc)[:80])]
        out = []
        for tr in data.get("triples", []):
            tag = str(tr.get("subject", "")) + "/" + str(tr.get("predicate", ""))
            try:
                c = Candidate(
                    subject=str(tr.get("subject", "")).strip(),
                    predicate=str(tr.get("predicate", "")).strip(),
                    new_predicate=str(tr.get("new_predicate", "")).strip(),
                    value=str(tr.get("value", "")).strip(),
                    start=int(tr.get("start", -1)),
                    end=int(tr.get("end", -1)),
                    tense=str(tr.get("tense", "current")).strip(),
                    multi=bool(tr.get("multi", False)),
                    conf=float(tr.get("conf", 0.0)),
                )
            except (TypeError, ValueError) as exc:
                dropped.append(("g1-schema", tag + " " + str(exc)[:60]))
                continue
            # G5 置信门槛
            if c.conf < self.MIN_CONF:
                dropped.append(("g5-conf", tag))
                continue
            # G2 时态门槛：过去/假设不写成 fact（原文已在 obs 里）；
            # hearsay 保留——存储层的 rel=2（听说）就是给它准备的
            if c.tense in ("past", "hypothetical"):
                dropped.append(("g2-tense:" + c.tense, tag))
                continue
            # G3.5 否定门槛：否定句没有断言新值（「不住在朝阳」≠ 住址=不住在朝阳）
            if re.match(r"^(不|没|没有|别|无|非|未)", c.value):
                dropped.append(("g3-5-negation", tag))
                continue
            ok, why = self._gate_subject(c, text)
            if not ok:
                dropped.append(("g3-subject:" + why, tag))
                continue
            ok, why = self._gate_value(c, text)
            if not ok:
                dropped.append(("g3-value:" + why, tag))
                continue
            ok, why = self._gate_predicate(c)
            if not ok:
                dropped.append(("g4-predicate:" + why, tag))
                continue
            out.append(c)
        # ---- 实体-实体关系（第二类输出）：关系也走闸门，写库后成为 rel 边 ----
        for rel in data.get("relations", []):
            tag = str(rel.get("from_entity", "")) + "-" + str(rel.get("relation", ""))
            try:
                frm = str(rel.get("from_entity", "")).strip()
                rname = str(rel.get("relation", "")).strip()
                to = str(rel.get("to_entity", "")).strip()
                conf = float(rel.get("conf", 0.0))
            except (TypeError, ValueError) as exc:
                dropped.append(("g1-schema", tag + " " + str(exc)[:60]))
                continue
            if conf < self.MIN_CONF:
                dropped.append(("g5-conf", tag))
                continue
            # 落地：两个实体名都要在原文里，或在别名表里（代词解析过的允许）
            def grounded(name):
                return bool(name) and (name in text or
                                       (self.extractor and self.extractor.canon(name)))
            if not (grounded(frm) and grounded(to)):
                dropped.append(("g6-relation-not-grounded", tag))
                continue
            if not rname or len(rname) > 6:
                dropped.append(("g6-relation-name", tag))
                continue
            out.append(Candidate(subject=frm, predicate="关系." + rname, value=to,
                                 multi=True, conf=conf, tense="current",
                                 is_relation=True))
        return out, dropped

    # ------------------------------------------------------------ 别名自学习
    def learn_entities(self, cands):
        """LLM 发现的新实体回写别名词典 → 下次正则快路直接命中。"""
        learned = []
        for c in cands:
            if c.is_new_entity and self.extractor is not None:
                if self.extractor.learn_alias(c.subject):
                    learned.append(c.subject)
        return learned


def default_registry():
    """与 DigitalHuman 初始谓词注册一致的默认白名单。"""
    return {p: {"multi": m} for p, m in
            (("电话", False), ("备用电话", True), ("工作单位", False),
             ("科室", False), ("住址", False), ("病情", False))}
