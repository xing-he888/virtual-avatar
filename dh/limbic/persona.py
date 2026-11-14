# -*- coding: utf-8 -*-
"""性格的自我塑造（第 7 步离线巩固的一部分）。

性格不是出厂定死的：config.PERSONA 只是「出厂种子」，真实性格 =
种子 + 经历的缓慢塑形。真值落在记忆库（calib 表 scope='persona'，
只插不改、取最新行——与遗忘常数同一套纪律）；data/persona.md 是
人可读的快照视图，不是真值。

三个可塑造的量：
    emo_mu  PAD 性格基线（p/a/d，钳 ±0.35）——近期经历极性以 0.1 的
            学习率把基线往自己拖；受骂变沉、被捧变亮，但极慢、有界。
            塑形后直接接进 OU 动力学的 mu——「平时的底色」真的移动。
    trust   对当前用户的关系态 0~1（初始 0.6「当朋友」）——被骂降、
            被夸升；进语气出口和评价语境。
    narrs   「我是谁」的自我叙事（self_narr 表，consolidate 已有）。

塑形只发生在第 7 步（空闲巩固）：证据不足（少于 3 条非中性经历）不动，
一次经历永远只挪一小步——性格不会一夜改变。
"""
import os
import time

import numpy as np

from .. import config

_MU_CLAMP = 0.35
_MU_EMA = 0.1           # 学习率：一次巩固最多把基线挪 10% 的差距
_TRUST_LEARN = 0.08
_TRUST_CLAMP = (0.15, 0.95)
_TRUST_INIT = 0.6
_MIN_EVIDENCE = 3       # 少于 3 条非中性经历不动性格（证据纪律）


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Persona:
    """小星的活性格：读、塑形、导出（MD 视图 + API 快照）。"""

    def __init__(self, dh):
        self.dh = dh

    # ------------------------------------------------ 读
    def mu(self):
        g = lambda k: self.dh.store.calib_latest("persona", "emo_mu", k)
        return (g("mu_p") or 0.0, g("mu_a") or 0.0, g("mu_d") or 0.0)

    def trust(self):
        v = self.dh.store.calib_latest("persona", "trust", "toward_user")
        return _TRUST_INIT if v is None else clamp(v, *_TRUST_CLAMP)

    def summary_text(self):
        """给 LLM 的一段「性格现状」：种子 + 塑形结果 + 自我叙事。"""
        p, a, d = self.mu()
        tr = self.trust()
        mood = "略偏阴沉" if p < -0.1 else ("明显开朗" if p > 0.15 else "平稳")
        stance = ("对用户有点戒备" if tr < 0.4 else
                  "和用户很亲近" if tr > 0.75 else "把用户当朋友")
        parts = ["出厂性格：" + config.PERSONA,
                 "经历塑形现状：情绪底色%s（P%+.2f）、%s（信任 %.2f）" % (
                     mood, p, stance, tr)]
        narrs = self.dh.store.self_narrs()[:3]
        if narrs:
            parts.append("自我认知：" + "；".join(r[1] for r in narrs))
        return "\n".join(parts)

    # ------------------------------------------------ 塑形（第 7 步）
    def shape(self, t=None):
        """用上次巩固以来的经历极性微调基线与信任。证据不足返回 None。"""
        dh = self.dh
        now = t if t is not None else time.time()
        since = dh.store.get_meta("persona_shaped_at")
        since = float(since) if since else 0.0
        rows = dh.store.con.execute(
            "SELECT valence, salience FROM obs WHERE valence IS NOT NULL "
            "AND valence != 0 AND t > ? AND (channel='appraisal' "
            "OR (kind='told' AND source != '小星')) ORDER BY t",
            (since,)).fetchall()
        # 编码强度**绝对**加权进塑形量：强烈经历对性格的影响更大（被狠戳一次
        # 记一辈子，顺嘴一句翻篇就忘）。系数 = clip(salience/BASE, 0, 3)——
        # 中性点位（0.15）恰为 1，与历史行为完全一致；最强 3 倍封顶
        # （性格不会一夜改变）；没编码的经历（gate≈0）几乎不塑形。
        base = config.EMO_SAL_BASE
        vals = [float(r["valence"]) *
                (1.0 if r["salience"] is None
                 else min(max(float(r["salience"]) / base, 0.0), 3.0))
                for r in rows]
        if len(vals) < _MIN_EVIDENCE:
            return None
        mean_p = sum(vals) / len(vals)
        # 基线：往经历极性挪一小步，钳住——性格极慢且有界
        p_old, a_old, d_old = self.mu()
        p_new = clamp(p_old * (1 - _MU_EMA) + mean_p * _MU_EMA,
                      -_MU_CLAMP, _MU_CLAMP)
        # 信任：正向交互升、负向降
        tr_new = clamp(self.trust() + _TRUST_LEARN * mean_p, *_TRUST_CLAMP)
        fitted = now
        for k, v in (("mu_p", p_new), ("mu_a", a_old), ("mu_d", d_old)):
            dh.store.calib_set("persona", "emo_mu", k, v, note="shape",
                               fitted_t=fitted)
        dh.store.calib_set("persona", "trust", "toward_user", tr_new,
                           note="shape", fitted_t=fitted)
        dh.store.set_meta("persona_shaped_at", repr(now))
        # 基线真正接进 OU 动力学的 mu——「平时的底色」移动，不只是标签
        dh.affect.dyn.mu = np.array([p_new, a_old, d_old])
        self.write_md(fitted)
        return dict(mu=(p_new, a_old, d_old), trust=tr_new,
                    n_evidence=len(vals), mean_valence=round(mean_p, 3))

    # ------------------------------------------------ 导出
    def md_path(self):
        """视图文件的落点：记忆库旁边。

        `:memory:` 库（demo、测试）**没有**视图文件——它们不该去覆盖磁盘上
        真实库的 data/persona.md（拿空 calib 覆盖真值快照，等于用假数据毁
        掉真数据）。注入的自定义库路径写到自己旁边，不污染项目内的视图。
        """
        dbp = getattr(self.dh.store, "path", ":memory:")
        if dbp in (None, "", ":memory:"):
            return None
        return os.path.join(os.path.dirname(os.path.abspath(dbp)), "persona.md")

    def write_md(self, fitted_t=None):
        """人可读快照：<记忆库同目录>/persona.md（视图，不是真值）。"""
        path = self.md_path()
        if path is None:
            return None                  # 内存库无视图可写（见 md_path）
        dh = self.dh
        p, a, d = self.mu()
        tr = self.trust()
        narrs = dh.store.self_narrs()
        last = dh.store.get_meta("persona_shaped_at")
        lines = [
            "# 小星的性格（自我塑造快照）",
            "",
            "> 真值在记忆库（%s）的 calib 表（scope=persona，只插不改）；"
            % os.path.basename(getattr(self.dh.store, "path", "") or "记忆库"),
            "> 本文件是人可读视图。出厂种子不会消失——塑形是在它上面的累积。",
            "",
            "## 出厂种子",
            "",
            config.PERSONA,
            "",
            "## 经历塑形现状",
            "",
            "- 情绪基线：P%+.2f / A%+.2f / D%+.2f（钳 ±%.2f）" % (p, a, d, _MU_CLAMP),
            "- 对当前用户的信任：%.2f（0=疏离，1=至交，出厂 %.2f）" % (tr, _TRUST_INIT),
            "- 最近一次塑形：%s" % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(
                    float(last))) if last else "还没有（空闲巩固时触发）"),
            "",
            "## 自我叙事（我是谁）",
            "",
        ]
        lines += ("- %s（权重 %.1f）" % (r[1], r[2]) for r in narrs) or \
                 ["- （还没有——自我叙事只在重复事件互相印证时生长）"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def snapshot(self):
        """API 视图：/api/persona。"""
        return dict(mu=self.mu(), trust=self.trust(),
                    seed=config.PERSONA,
                    narrs=[r[1] for r in self.dh.store.self_narrs()],
                    last_shaped=self.dh.store.get_meta("persona_shaped_at"),
                    md=self.md_path())
