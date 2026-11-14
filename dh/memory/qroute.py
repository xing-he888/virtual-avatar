# -*- coding: utf-8 -*-
"""问句路由器（semantic router 的 L0/L1 层；L2 的 LLM 兜底在 llm_bridge）。

为什么需要（本会话实测）：手工问句小抄（rules._QUESTION_PRED，6 条）是开放
词表上的枚举——「年龄」这类 LLM 新谓词永远不在表里，「我多少岁」只能整包
丢给大模型现想（实测 4 问错 2，且来源分级措辞失效）。业界标准解法是
semantic router：预嵌入每个路由的示例问法 → 来句嵌入 → 余弦最近邻 → 命中
才点查，模糊才劳驾 LLM；LLM 解析成功的新问法回填线索库——**表自己生长，
零人工**（这正是实体别名词典「LLM 发现新实体回写 → 正则快路下次直接命中」
的同一条自学习回路，复制到谓词上）。

三层分工：
    L0 确定性   「我」→属主（写侧 _deref_pronouns 的读侧半边，一条规则管
                所有自指问句）+ 线索精确/子串 + 谓词名命中
    L1 语义     问句嵌入 vs 线索嵌入 ≥QROUTE_MIN_COS（本机实测：正确路由
                0.73-0.89 / 错配 ≤0.53 / 无关 ≤0.52，0.6 居中两侧余量 ≥0.13）；
                问句向量召回反正要算，这里复用缓存，零额外成本
    L2 LLM 兜底 解析合法即回填线索（答没答得上另说）→ 下次 L1 直达
                （llm_bridge.llm_route_question，纪律与抽取五道闸门同款：
                 谓词必须已注册、主体必须落地）

存储：pred_hint 表（真值，只增，UNIQUE(谓词,问法)）+ vec 表 kind='pred_hint'
（向量真值）——与 alias/calib/value_alias 同一模式；运行时全内存
（小字典 + 几 KB 小矩阵）。hash 嵌入口径下语义层停用（阈值没标定过，
同 equiv 纪律），子串/谓词名层照常。
"""
import numpy as np

from .. import config
from ..extract.rules import has_kinship_term
from . import embed as embed_mod

# 六个出厂谓词的种子问法（bootstrap，一次性手写；之后全部自学习）
SEED_HINTS = {
    "电话": ["电话号码是多少", "联系方式是什么"],
    "备用电话": ["备用电话是多少"],
    "工作单位": ["在哪个单位上班", "在哪里工作"],
    "科室": ["在哪个科室"],
    "住址": ["住在哪里", "家在什么地方"],
    "病情": ["身体怎么样了", "病情如何"],
}


class QuestionRouter:
    def __init__(self, store, embed_fn, extractor=None):
        self.store = store
        self.embed_fn = embed_fn
        self.ex = extractor            # 别名词典/属主消解借用（Writer 的 Extractor）
        self.cos_enabled = getattr(embed_fn, "model_name", None) == embed_mod.MODEL_NAME
        self._cache = None             # (model, hints, pred_by_id, ids, mat)
        self.refresh()

    # ------------------------------------------------ 线索库
    def refresh(self):
        model = getattr(self.embed_fn, "model_name", None) or "hash256"
        hints = [(r["hint_id"], r["predicate"], r["question"])
                 for r in self.store.pred_hints_all()]
        pred_by_id = {int(r["hint_id"]): r["predicate"]
                      for r in self.store.pred_hints_all()}
        ids, mat = self.store.load_matrix("pred_hint", model=model)
        self._cache = (model, hints, pred_by_id, ids, mat)

    def learn(self, predicate, question, t=0.0, source="llm"):
        """登记一条问法线索（幂等）+ 向量入库 + 热缓存刷新。返回 hint_id 或 None。"""
        q = (question or "").strip()
        if not predicate or not q or len(q) > 40:
            return None
        hid = self.store.pred_hint_learn(predicate, q, t=t, source=source)
        if hid is None:
            return None
        if self.cos_enabled:
            self.store.put_vec("pred_hint", hid, self.embed_fn(q),
                               model=getattr(self.embed_fn, "model_name", "hash256"))
        self.refresh()
        return hid

    def forget(self, question):
        """撤销一条线索（误教清理用；UI 未接，管理面手动调）。"""
        self.store.con.execute("DELETE FROM pred_hint WHERE question=?",
                               (question,))
        self.store.con.execute(
            "DELETE FROM vec WHERE kind='pred_hint' AND row_id NOT IN "
            "(SELECT hint_id FROM pred_hint)")
        self.refresh()

    def seed(self, t=0.0):
        """出厂种子线索（幂等，启动时调一次）。返回新种条数。"""
        n = 0
        for pred, qs in SEED_HINTS.items():
            for q in qs:
                if self.learn(pred, q, t=t, source="seed") is not None:
                    n += 1
        return n

    # ------------------------------------------------ 路由
    def route(self, text):
        """问句 → 点查参数 dict（与 parse_question 同形，多一个 via 字段），
        或 None（路由不出来 → 交给 RAG / L2）。

        ⚠ 注意：parse_question 命中时**不区分陈述/问句**（旧 ask 同口径），
        调用方（policy/chat_service）自行用 looks_like_question 把门。"""
        parsed = self.ex.parse_question(text) if self.ex else None
        if parsed is not None:
            parsed.setdefault("via", "t0")
            return parsed
        from ..core.tick import looks_like_question   # 惰性导入：tick 载入本模块，避免环
        if not looks_like_question(text):
            return None
        # 频次/元问题不路由：「换了多少次」「几个号」问的是记忆里没有的统计，
        # 点查只会答非所问（实测「我电话号码换了多少次了」曾自信错答号码）
        if any(w in text for w in ("多少次", "几次", "几回")):
            return None
        # 剥「听X说，」前缀：「我」指说话人本人，不指传话人（与 t0 路同口径）
        _src, body = self.ex.split_source(text) if self.ex else (None, text)
        # L0：实体（别名）→「我」→属主。句中带亲属/关系称呼时「我」不指
        # 本人（我姐姐的电话≠我的电话），不得消解到属主——错答比不答更糟
        ents = self.ex.find_entities(body) if self.ex else []
        owner = self.store.get_owner()
        if ents:
            subject = ents[0][2]
        elif (owner and "我" in body
              and not has_kinship_term(body)):
            subject = owner
            ents = [(max(body.find("我"), 0), "我", owner)]
        else:
            return None
        # 线索：子串精确（学过的问法常被原样重复 / 谓词名直呼）→ 语义
        pred = self._hint_substring(body)
        if pred is None and self.cos_enabled:
            pred = self._hint_semantic(text)
        if pred is None:
            return None
        return dict(src=None, subject=subject, predicate=pred, value=None,
                    entities=ents, change=False, body=body, via="hint")

    # ------------------------------------------------ 内部
    def _hint_substring(self, text):
        for _hid, pred, q in self._cache[1]:
            if q in text or (len(pred) <= 4 and pred in text):
                return pred
        return None

    def _hint_semantic(self, text):
        model, hints, pred_by_id, ids, mat = self._cache
        if not hints or mat is None or not mat.size:
            return None
        qv = self.embed_fn(text)
        qn = float((qv * qv).sum()) ** 0.5
        if qn <= 1e-9 or qv.size != mat.shape[1]:
            return None
        with np.errstate(all="ignore"):
            sims = mat @ (qv / qn)
        best = int(np.argmax(sims))
        if float(sims[best]) < config.QROUTE_MIN_COS:
            return None
        # 用 ids 映射回谓词——hints 按表序、矩阵按 row_id 序，缺向量行的
        # 线索（hash 口径学习/put_vec 异常被吞）会让两序列错位，按位置取
        # 会路由到错误谓词（独立检验实证过：精确匹配却答错人的字段）
        return pred_by_id.get(int(ids[best]))
