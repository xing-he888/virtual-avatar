# -*- coding: utf-8 -*-
"""第 7 步（离线）：巩固 + 心智游移 [DESIGN §5.2 / 期 5]。

巩固做什么：
    - 把「同一件事」的说法聚到事件牌子上（写入时已做），离线时给多说法事件
      补齐槽位总结；
    - 「我是谁」的自我叙事只在这里改，且极慢（需强烈/重复事件）；
    - 「性格」也只在这里被经历塑形（persona.shape：情绪基线/关系信任，
      极慢、有界、证据不足不动）；
    - 「遗忘」不删任何行——情节流只追加 [DESIGN 规则 2]，遗忘发生在检索侧
      （时间核权重），这是设计不是偷懒。

心智游移做什么：
    空闲时随机抽几段情节做联想（ρ=0.55 / K=200 / 抽样，[GRAPH §9.5]），
    产出不一定说出来，但会写回记忆（rel=3，L3 联想——只能说「让我想起…」，
    不能当事实陈述 [DESIGN §6 第一层]）。
"""
from .. import config
from ..limbic.persona import Persona


class Offline:
    def __init__(self, digital_human):
        self.dh = digital_human

    # ------------------------------------------------ 巩固
    def consolidate(self, t, min_mentions=3):
        """返回更新的自我叙事条数。性格塑形在收尾时顺带完成（persona.shape）。"""
        store = self.dh.store
        made = 0
        evs = store.con.execute(
            "SELECT event_id,label,key,t_start,t_end FROM event").fetchall()
        for ev in evs:
            n = store.con.execute(
                "SELECT COUNT(*) FROM mention WHERE event_id=?",
                (ev["event_id"],)).fetchone()[0]
            if n < min_mentions:
                continue
            if ev["key"] and ev["key"].startswith("小星|"):
                row = store.con.execute(
                    "SELECT f.value FROM fact f JOIN mention m ON m.obs_id=f.src_obs "
                    "WHERE m.event_id=? AND f.status != 'retracted' LIMIT 1",
                    (ev["event_id"],)).fetchone()
                if row is None:
                    continue
                pred = ev["key"].split("|")[1]
                claim = "我的%s是%s（%d 种说法互相印证）" % (pred, row["value"], n)
                if not any(claim[:12] == r[1][:12] for r in store.self_narrs()):
                    store.add_self_narr(claim, 0.6 + 0.1 * n, t)
                    made += 1
        # 性格塑形：经历极性 -> 情绪基线/关系信任（极慢、有界、证据不足不动）
        try:
            Persona(self.dh).shape(t)
        except Exception:                       # noqa: BLE001
            pass                                # 塑形失败不影响巩固本身
        return made

    def _wander_seed_facts(self, k=3):
        """游移的兜底种子：优先编码强度高的经历。

        人放空时更爱回味情绪重的事——种子按 obs.salience（写入时右脑 gate
        落的）降序取，中性行（0.5）退回按时间。"""
        return [int(r[0]) for r in self.dh.store.con.execute(
            "SELECT f.fact_id FROM fact f JOIN obs o ON o.obs_id=f.src_obs "
            "WHERE f.status != 'retracted' "
            "ORDER BY COALESCE(o.salience, 0.15) DESC, f.written_t DESC LIMIT ?",
            (k,))]

    # ------------------------------------------------ 心智游移
    def wander(self, t, rng, rounds=None, per_round=None, say_seed=None):
        """空闲触发。种子 = 最近进工作记忆的事 + 当前心情。

        返回 dict(thoughts=[联想出的话], rel=3)。写回 obs(kind='infer', rel=3)。
        """
        from ..limbic.emotion import PAD
        rounds = config.ASSOC_ROUNDS if rounds is None else rounds
        per_round = min(config.ASSOC_PER_ROUND if per_round is None else per_round,
                        6)     # 游移是轻量的，不像主动回忆开满 20 条
        dh = self.dh
        seeds = {}
        for it in dh.wm.items[-3:]:
            fid = it.get("fact_id") if isinstance(it, dict) else None
            if fid is not None:
                seeds[fid] = 1.0
        if not seeds:
            seeds = {fid: 1.0 for fid in self._wander_seed_facts()}
        if not seeds:
            return dict(thoughts=[])
        res = dh.graph.associate(seeds, rng, rounds=rounds, per_round=per_round)
        thoughts = []
        for picks in res["picks"]:
            for fid in picks[:1]:
                row = dh.store.fact_row(fid)
                if row is None:
                    continue
                obs = dh.store.obs_row(row["src_obs"])
                text = obs["payload"] if obs is not None else ""
                # 小记忆库里联想会打转（节点太少），游移的话别重复
                if text and text not in thoughts:
                    thoughts.append(text)
        if thoughts:
            # 写回：L3 联想，kind=infer；只能以「让我想起」的口吻说，不当事实
            dh.store.add_obs(t, "infer", "mind", "｜".join(thoughts[:2]),
                             source="mind", rel=3, seg_bnd=0.0)
        return dict(thoughts=thoughts, touched=res["touched"],
                    repeat_rate=res["repeat_rate"], seen=res.get("seen", {}))
