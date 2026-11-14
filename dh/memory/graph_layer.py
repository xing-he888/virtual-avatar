# -*- coding: utf-8 -*-
"""激活层：节点 + 三类边 + 扩散激活，回答「提到老王，还有什么会跟着亮」。

它不是另一份数据，是真值层上的一层索引 [GRAPH §0]：CSR 矩阵从
fact / event / vec 三张表派生（永远只有一个真值来源），外加一张
link 边表存人/LLM 手工打的边（那才是图里不能自动推导的信息）。

边三种，自动建 92% [GRAPH §1]：
    link  人/LLM 写入时打的边            w=1.0
    share 共享实体（纯规则）              w=0.7
    sim   向量近邻（余弦超阈值就连）       w=0.5

扩散公式 [GRAPH §3.1]：
    h_0 = 种子（归一化）
    h_{t+1} = decay · (A / sqrt(deg)) · h_t
    激活值 = Σ_t h_t        每一跳都累加，不是只取最后一跳
    decay=0.55，3 跳最好（5 跳反而降）；PPR 在枢纽多的小图上更差（0.52），不用。

枢纽压制（设计核心约束 [GRAPH §4]）：
    - share 边按实体规模摊薄：同一实体名下 n 条记忆，边权 ×min(1, D/n)
    - sqrt(deg) 归一化
    - 实体节点不进扩散矩阵（它的边只用来进入；「这个人的记忆」用 SQL 一跳
      显式展开，见 activate_for_entities）——这是 §4 第 1/3 条的落地方式

联想循环 [GRAPH §9.5 / E48c/E48e]：
    ρ=0.55 已想起降权、K=200 点亮筐、每轮按分数抽样 20 条（不是取最大——
    argmax 下筐大小对结果毫无影响，数学上必然）。
    不做跨人「情绪相近」边：实测一个位置都没多出来，只把跑偏换成别人的坏消息。
"""
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix

from .. import config


class ActivationGraph:
    def __init__(self, store):
        self.store = store
        self._edge_version = -1
        self._ids = None              # fact_id -> 矩阵行号
        self._acc = None              # 累积邻接（未归一化），csr
        self._an = None               # sqrt 归一化后的传播矩阵
        self._dirty = True

    # ------------------------------------------------ 派生与重建
    def refresh(self, force=False):
        """边由真值层派生；有新边才重建 CSR（联想只在离线步跑，不进热路径）。"""
        n_edges = self.store.con.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
        n_fact = self.store.fact_count()
        version = (n_edges, n_fact)
        if not force and not self._dirty and version == self._edge_version:
            return
        edges = self._derive_edges()
        ids = sorted({e[0] for e in edges} | {e[1] for e in edges})
        idx = {fid: i for i, fid in enumerate(ids)}
        n = len(ids)
        rows, cols, vals = [], [], []
        for a, b, w in edges:
            if a in idx and b in idx:
                rows.append(idx[a]); cols.append(idx[b]); vals.append(w)
        if n == 0:
            self._ids, self._acc, self._an = np.empty(0, np.int64), None, None
        else:
            acc = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
            acc.setdiag(0)
            acc.sum_duplicates()
            deg = np.asarray(acc.sum(axis=1)).ravel()
            deg[deg == 0] = 1.0
            # 对称化度归一：A/√deg 出 / √deg 入（demo_chain 的写法）
            inv = 1.0 / np.sqrt(deg)
            an = acc.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
            self._ids, self._acc, self._an = np.array(ids, np.int64), acc, an
        self._edge_version = version
        self._dirty = False

    def _derive_edges(self):
        """share（同实体，按规模摊薄）+ link（同事件）+ sim（向量近邻）+ 手工 link。"""
        out = set()
        # share：同实体的记忆两两相连，边权按实体名下条数摊薄（压枢纽）
        rows = self.store.con.execute(
            "SELECT fact_id, entity FROM fact "
            "WHERE status != 'retracted' ORDER BY fact_id").fetchall()
        grp = defaultdict(list)
        for r in rows:
            grp[r["entity"]].append(r["fact_id"])
        for ent, fids in grp.items():
            w = config.EDGE_W["share"] * min(1.0, config.HUB_DEG_CAP / len(fids))
            for i in range(len(fids)):
                for j in range(i + 1, len(fids)):
                    out.add((fids[i], fids[j], w))
                    out.add((fids[j], fids[i], w))
        # link：同一块事件牌子下的说法（E38：零共同词时只有事件节点救得回来）
        ev_rows = self.store.con.execute(
            "SELECT m1.event_id AS e, m1.mention_id AS ma, m2.mention_id AS mb, "
            "o1.obs_id AS oa, o2.obs_id AS ob FROM mention m1 "
            "JOIN mention m2 ON m2.event_id=m1.event_id AND m2.mention_id>m1.mention_id "
            "JOIN obs o1 ON o1.obs_id=m1.obs_id JOIN obs o2 ON o2.obs_id=m2.obs_id"
        ).fetchall()
        obs2fact = {}
        fr = self.store.con.execute(
            "SELECT fact_id, src_obs FROM fact "
            "WHERE status != 'retracted'").fetchall()
        for r in fr:
            obs2fact[r["src_obs"]] = r["fact_id"]
        for r in ev_rows:
            a, b = obs2fact.get(r["oa"]), obs2fact.get(r["ob"])
            if a is not None and b is not None and a != b:
                out.add((a, b, config.EDGE_W["link"]))
                out.add((b, a, config.EDGE_W["link"]))
        # sim：向量近邻（cos >= 阈值就连）。规模小，全对一次算完
        ids, mat = self.store.load_matrix("fact")
        if len(ids) >= 2 and mat.shape[1] > 0:
            nrm = np.linalg.norm(mat, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            m = mat / nrm
            # errstate：macOS Accelerate BLAS 跑完 m @ m.T 会留下虚假的
            # divide/overflow 浮点标志，numpy 1.26 把它误报成 RuntimeWarning
            # ——实测结果矩阵无 NaN/inf（float32/float64 都报），压掉报告即可
            with np.errstate(all="ignore"):
                sim = m @ m.T
            ii, jj = np.nonzero(sim >= config.SPREAD_SIM_MIN_COS)
            for a, b in zip(ii, jj):
                if a < b:
                    out.add((int(ids[a]), int(ids[b]),
                             config.EDGE_W["sim"] * float(sim[a, b])))
                    out.add((int(ids[b]), int(ids[a]),
                             config.EDGE_W["sim"] * float(sim[a, b])))
        # 手工 link（人/LLM 打的；node/edge 表只服务这一种）
        for a, b, kind, w in self.store.edges_all():
            if kind == "link":
                out.add((a, b, w))
                out.add((b, a, w))
        return out

    # ------------------------------------------------ 扩散
    def diffuse(self, seed_scores, hops=None, decay=None):
        """seed_scores: {fact_id: 强度}。返回 {fact_id: 激活值}。

        图外孤星种子（没有边的事实）至少点亮自己——种子进不了扩散矩阵
        不等于「想不起来」，只等于「扩不出去」。"""
        self.refresh()
        if not seed_scores:
            return {}
        if self._an is None:
            tot = sum(seed_scores.values()) or 1.0     # 图上无边：孤星直接亮
            return {int(f): float(s) / tot for f, s in seed_scores.items()}
        hops = config.DIFF_HOPS if hops is None else hops
        decay = config.DIFF_DECAY if decay is None else decay
        idx = {int(f): i for i, f in enumerate(self._ids)}
        h = np.zeros(len(self._ids))
        tot = sum(seed_scores.values()) or 1.0
        for fid, s in seed_scores.items():
            if int(fid) in idx:
                h[idx[int(fid)]] = s / tot
        acc = h.copy()
        cur = h
        for _ in range(hops):
            cur = decay * (self._an @ cur)
            acc = acc + cur
        res = {int(self._ids[i]): float(acc[i]) for i in np.nonzero(acc)[0]}
        # 图外孤星（没有边的记忆）：至少点亮自己——否则单条记忆的实体
        # 在扩散里完全隐形
        for fid, s in seed_scores.items():
            if int(fid) not in res:
                res[int(fid)] = res.get(int(fid), 0.0) + float(s)
        return res

    def activate_for_entities(self, entities, predicate=None, extra=0.0):
        """「问老王电话」不该从老王出发扩散 [GRAPH §4 第 3 条]：
        先按实体（+谓词）用 SQL 挑出记忆集合，再以它们为种子扩散。
        关系事实（关系.X=对方）把关联实体并进种子集合——提老王，
        老王儿子的记忆也会亮。"""
        ents = list(dict.fromkeys(entities))
        # 关系扩展：双向查（我拥有「关系.X=对方」+ 对方拥有「关系.X=我」）
        ph = ",".join("?" * len(ents))
        rows_a = self.store.con.execute(
            "SELECT DISTINCT value FROM fact WHERE entity IN (%s) "
            "AND predicate LIKE '关系.%%' AND status != 'retracted'" % ph, ents).fetchall()
        rows_b = self.store.con.execute(
            "SELECT DISTINCT entity FROM fact WHERE value IN (%s) "
            "AND predicate LIKE '关系.%%' AND status != 'retracted'" % ph, ents).fetchall()
        for r in rows_a:
            if r["value"]:
                ents.append(r["value"])
        for r in rows_b:
            if r["entity"]:
                ents.append(r["entity"])
        ents = list(dict.fromkeys(ents))
        q = "SELECT fact_id FROM fact WHERE entity IN (%s) " % (
            ",".join("?" * len(ents)),)
        q += "AND status != 'retracted'"
        args = list(ents)
        if predicate:
            q += " AND predicate=?"
            args.append(predicate)
        fids = [r[0] for r in self.store.con.execute(q, args).fetchall()]
        return self.diffuse({f: 1.0 + extra for f in fids})

    # ------------------------------------------------ 联想循环
    def associate(self, seed_scores, rng, rounds=None, per_round=None,
                  basket=None, rho=None, hops=None, decay=None, how="sample"):
        """连续联想 [GRAPH §9]：上一轮讲出来的事做下一轮的种子。

        how="sample" 按分数无放回抽样（默认，文档口径）；
        how="argmax" 取分数前 per_round 条（E48e 的对照臂——
        它对筐大小不敏感：top-20 由分数排序决定，筐再大也只取前 20）。

        返回 dict(picks=[每轮选中的 fact_id 列表], touched=累计去重数,
                  repeat_rate=重复率, seen={fact_id: 被想起次数})。
        """
        rounds = config.ASSOC_ROUNDS if rounds is None else rounds
        per_round = config.ASSOC_PER_ROUND if per_round is None else per_round
        basket = config.ASSOC_BASKET if basket is None else basket
        rho = config.SEEN_RHO if rho is None else rho
        self.refresh()
        if not seed_scores:
            return dict(picks=[], touched=0, repeat_rate=0.0, seen={})
        idx = {int(f): i for i, f in enumerate(self._ids)}
        # 种子 = 图上节点 + 图外孤悬（diffuse 的孤星兜底会点亮后者）。
        # 只认图内种子的话，一条没边的关系事实会让整个联想空手（实测触达 0）
        sv = {int(f): float(s) for f, s in seed_scores.items()}
        all_ids = [int(f) for f in self._ids] + [f for f in sv if f not in idx]
        pos = {f: i for i, f in enumerate(all_ids)}
        seen = {}                    # 纯 dict：查询不插入（touched 才不会被污染）
        picks_all = []
        for _ in range(rounds):
            acc = self.diffuse(sv, hops=hops, decay=decay)
            if not acc:
                break
            a = np.zeros(len(all_ids))
            for fid, v in acc.items():
                a[pos[int(fid)]] = v
            a = a * (rho ** np.array([seen.get(f, 0) for f in all_ids]))
            order = np.argsort(-a)
            box = order[:basket]
            k = min(per_round, len(box))
            if how == "argmax":
                pick = box[:k]
            else:
                w = np.maximum(a[box], 1e-12)
                w = w / w.sum()
                pick = rng.choice(box, size=k, replace=False, p=w)
            picks = [int(all_ids[i]) for i in pick]
            picks_all.append(picks)
            for f in picks:
                seen[f] = seen.get(f, 0) + 1
            sv = {int(all_ids[i]): 1.0 for i in pick[:3]}
        touched = len(seen)
        n_pos = rounds * per_round
        return dict(picks=picks_all, touched=touched,
                    repeat_rate=round(1.0 - touched / max(n_pos, 1), 4),
                    seen=dict(seen))
