# -*- coding: utf-8 -*-
"""写入侧的值等价仲裁器：「这句话的值，是不是我们已经记着的那件事的另一种说法？」

为什么需要（本会话实测，见 config.WRITE_EQUIV_COS 的出处注释）：
    LLM Tier 1 的值逐字落地（G3），同一事实的两种表面形式（海淀/海淀区）
    会写成两行 fact：单值谓词假变更（旧行被 superseded、valid_from 重置，
    时间旅行里多出一次没发生的搬家）、多值谓词并存、标定流 phi 虚高
    （翻转率被表面形式搅动污染，遗忘常数 S 被系统性低估）。

裁决顺序（每一步都是「确定性优先」）：
    1 词典归一      rules.normalize_value（进医院→住院）——Tier 0 与 Tier 1 同一张表
    2 规范值精确命中  归一后与某 active 行逐字相等 -> 直接确认（零余弦成本）
    3 别名表        value_alias 里记过的表面形式 -> 微秒命中
    4 豁免          数字类谓词 / 关系谓词 / 数字串值 / 过短值 —— 不比
    5 value-only 余弦  与该 (实体,谓词) 的全部 active 值比，>=WRITE_EQUIV_COS 确认

三条纪律：
    * 只看 active 行——复述已失效的值是「传旧话」，由 write_fact 的
      confirm_old 分支管，等价确认绝不复活它。
    * 合并不改写：fact.value 永远是先到的规范值，新表面形式只进 value_alias
      和 obs 原文。「值逐字落地」纪律不破。
    * 嵌入模型不是 bge 时整个仲裁器停用（hash 嵌入没有标定过的阈值，
      subset 包含会把「海淀」拉到危险的高分）——宁可回到旧行为。
"""
from .. import config
from ..extract.rules import normalize_value
from . import embed as embed_mod


def is_numeric_value(value):
    """数字串值不比语义（两个电话号余弦再高也可能真不同）。"""
    digits = sum(ch.isdigit() for ch in value)
    return digits > 0 and digits >= 0.5 * max(len(value), 1)


class ValueEquivalence:
    """挂在 Writer 上（writer 持有 embed_fn）。find() 返回 (fact_id, 规范值) 或 None。"""

    def __init__(self, store, embed_fn):
        self.store = store
        self.embed_fn = embed_fn
        # 只有标定过的 bge 口径才启用余弦；hash 兜底只享受词典/别名两层
        self.cos_enabled = getattr(embed_fn, "model_name", None) == embed_mod.MODEL_NAME

    # ------------------------------------------------ 主入口
    def find(self, entity, predicate, value):
        """该值是否等价于 (entity, predicate) 名下某个 active 值。

        返回 (fact_id, canonical)；不是等价复述返回 None（调用方照常走
        write_fact 的 new/confirm/confirm_old）。

        边界（实测踩过）：字面完全相同的复述**不在这里截**——那要留给
        write_fact 的 confirm 分支（动作名与「传旧话/复活」的判定是它的）；
        这里只裁「字面不同但语义相同」的部分。余弦循环也跳过与查询串
        完全相同的候选，否则自己跟自己比出 1.0，永远命中。"""
        raw = (value or "").strip()
        v = normalize_value(predicate, raw)
        if not v:
            return None

        # 2 词典等价：归一改写了值（进医院→住院）且规范值 active。
        #    归一没改写（v==raw）时不截——那是字面命中，write_fact 自己会 confirm
        if v != raw:
            row = self._active_row(entity, predicate, v)
            if row is not None:
                return int(row["fact_id"]), v

        # 3 别名表：这个表面形式以前被裁过等价
        hit = self.store.value_alias_lookup(entity, predicate, v)
        if hit is not None:
            canon = hit["canonical"]
            if canon == v:
                return None          # 别名指向的规范值 == 本串：字面命中，交给 write_fact
            row = self._active_row(entity, predicate, canon)
            if row is not None:
                return int(row["fact_id"]), canon
            # 规范行已失效：若本串字面命中某条失效行，write_fact 会给 confirm_old；
            # 字面不命中则落 new（复述失效值的别名形态不复活——传旧话纪律）
            return None

        # 4 豁免：不比
        if (predicate in config.WRITE_EQUIV_EXEMPT_PREDS
                or predicate.startswith(config.WRITE_EQUIV_EXEMPT_PREFIX)
                or is_numeric_value(v)
                or len(v) < config.WRITE_EQUIV_MIN_LEN):
            return None

        # 5 value-only 余弦（只在 bge 口径下）
        actives = self.store.active_values(entity, predicate)
        if not actives or not self.cos_enabled:
            return None
        try:
            qv = self.embed_fn(v)
        except Exception:            # noqa: BLE001  嵌入坏了不能拖垮写入
            return None
        qn = float((qv * qv).sum()) ** 0.5
        if qn <= 1e-9:
            return None
        best = None
        for r in actives:
            stored = r["value"]
            if stored == raw:
                continue             # 字面完全相同的复述归 write_fact 的 confirm 分支
            cv = normalize_value(predicate, stored)
            if cv == v:
                # 归一后同串（存侧是未归一的 raw、来侧已归一，或反之）：
                # 词典等价的双向形态，余弦必为 1.0，直接合并
                return int(r["fact_id"]), stored
            # 确定性前缀否决：一方是另一方的前缀且长度差 >=2，字面多出来的
            # 部分可能就是信息（协和医院/协和医院西院 cos=0.862，不能合）。
            # 海淀区/海淀 差 1 个「区」字不否决——那正是要接住的形态差。
            if ((cv.startswith(v) or v.startswith(cv))
                    and abs(len(cv) - len(v)) >= 2):
                continue
            cv_vec = self.embed_fn(cv)
            cn = float((cv_vec * cv_vec).sum()) ** 0.5
            if cn <= 1e-9 or cv_vec.size != qv.size:
                continue
            sim = float(qv @ cv_vec) / (qn * cn)
            if sim >= config.WRITE_EQUIV_COS and (best is None or sim > best[0]):
                best = (sim, int(r["fact_id"]), stored)
        if best is None:
            return None
        return best[1], best[2]

    # ------------------------------------------------ 内部
    def _active_row(self, entity, predicate, value):
        return self.store.con.execute(
            "SELECT fact_id,value FROM fact "
            "WHERE entity=? AND predicate=? AND value=? AND status='active' "
            "ORDER BY fact_id DESC LIMIT 1", (entity, predicate, value)).fetchone()
