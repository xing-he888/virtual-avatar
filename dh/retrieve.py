# -*- coding: utf-8 -*-
"""检索层。三种问题走三条路，LLM 只在最后一步出现 [GRAPH §5]：

    「老王电话多少」        点查（entity+predicate+status 索引）   微秒级
    「这个人我刚才见过吗」   语义层 top-k                          毫秒级
    「提到老王我还知道什么」 激活层 3 跳扩散                       毫秒级
    复杂的                  把上面三者的结果塞进上下文，让 LLM 组织语言

幻觉治理两层在这里落地 [DESIGN §6]：
    来源分级 L0-L4 -> 措辞（我看到 / 你告诉过我 / 听X说 / 我猜）
    过期检测       -> 引用失效事实时降级为「我记得以前是…现在不确定」
    弃权           -> 放最后做，而且不要期待它（实测只 +0.001）
"""
import numpy as np

from . import config, kernel
from .scoring import hybrid_score


class Answer:
    """一次事实问答的完整出处。is_stale/is_guess 决定措辞，value 是内容。"""

    def __init__(self, values, rel=None, sources=None, stale=False, abstain=False,
                 degraded=False, fact_ids=None):
        self.values = values            # list[str]
        self.rel = rel                  # 来源分级（最高可信的那个）
        self.sources = sources or []    # 最早来源（谁最早说的）
        self.stale = stale              # 全部支撑已失效
        self.abstain = abstain
        self.degraded = degraded
        self.fact_ids = fact_ids or []  # 支撑本回答的事实行（动态修正锚定用）

    def phrase(self):
        if self.abstain or not self.values:
            return config.ABSTAIN_PHRASE
        vals = "和".join(self.values)
        if self.stale:
            return config.EXPIRED_PHRASE.format(old=vals)
        if self.rel == 0:
            return "我看到是%s" % vals
        if self.rel == 1:
            return "你告诉过我，是%s" % vals
        if self.rel == 2 and self.sources:
            return "听%s说，是%s" % (self.sources[0], vals)
        if self.rel == 3:
            return "我猜是%s" % vals
        return "是%s" % vals


class Retriever:
    def __init__(self, store, embed_fn=None, S=None, sigma=None):
        self.store = store
        self.embed_fn = embed_fn
        self.S = S
        self.sigma = sigma

    # ------------------------------------------------ 点查（事实）
    def ask_fact(self, entity, predicate, asof=None):
        """「老王现在电话多少」。双时态存储让「现在」只是 valid_to IS NULL 的特例。

        asof 给定时走区间查询（valid_from/valid_to 是世界时间，status 是
        数据库时间——回看历史必须用区间，不能信 status）。
        多值谓词返回全部 active 行；单值谓词一行。
        只有失效行时 -> degraded（「我记得以前是…」），绝不当现在的事实端出来
        [DESIGN §6 第二层]。
        """
        if asof is None:
            rows = self.store.active_values(entity, predicate)
        else:
            rows = self.store.value_at(entity, predicate, asof)
        if rows:
            rel = min(r["rel"] for r in rows)
            srcs = self._sources(rows)
            return Answer([r["value"] for r in rows], rel=rel, sources=srcs,
                          fact_ids=[r["fact_id"] for r in rows])
        old = self.store.superseded_values(entity, predicate)
        if old:
            return Answer([old[0][0]], rel=None, stale=True, degraded=True)
        return Answer([], abstain=True)

    def _sources(self, rows):
        """来源留最早那条 [extract.py 注释 2]。"""
        srcs = []
        for r in rows:
            obs = self.store.obs_row(r["src_obs"])
            if obs is not None and obs["source"]:
                srcs.append(obs["source"])
        return srcs

    # ------------------------------------------------ 时间旅行
    def ask_fact_at(self, entity, predicate, t):
        rows = self.store.value_at(entity, predicate, t)
        if not rows:
            return Answer([], abstain=True)
        return Answer([r[1] for r in rows], rel=1)

    # ------------------------------------------------ 语义路
    def semantic_topk(self, query_vec, kind="fact", k=8, mask_ids=None):
        """numpy 暴力扫。20 万条以内完全够 [STORE §4]。"""
        if self.embed_fn is None or not callable(self.embed_fn):
            return []
        ids, mat = self._mat(kind)
        if len(ids) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        if q.size != mat.shape[1]:
            return []
        sims = mat @ (q / (np.linalg.norm(q) + 1e-9))
        if mask_ids is not None:
            allow = np.zeros(len(ids), dtype=bool)
            pos = {int(i): j for j, i in enumerate(ids)}
            for i in mask_ids:
                if int(i) in pos:
                    allow[pos[int(i)]] = True
            sims = np.where(allow, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [(int(ids[i]), float(sims[i])) for i in order if np.isfinite(sims[i])]

    def _mat(self, kind):
        """向量矩阵缓存：行数没变就复用，变了才重读（小规模下都便宜）。"""
        if getattr(self, "_mat_cache", None) is None:
            self._mat_cache = {}
        ids, mat = self.store.load_matrix(kind)
        cached = self._mat_cache.get(kind)
        if cached is None or len(cached[0]) != len(ids):
            self._mat_cache[kind] = (ids, mat)
        return self._mat_cache[kind]

    # ------------------------------------------------ 三路混合召回（「讲点什么」用）
    def _pool_recent(self, t_now, n=30):
        """时间路候选（事实层）：按时间核权重取最近 n 条（「上次」「那天」）。"""
        rows = self.store.con.execute(
            "SELECT fact_id, written_t FROM fact "
            "WHERE status != 'retracted' ORDER BY written_t DESC LIMIT ?",
            (n * 3,)).fetchall()
        if not rows or t_now is None:
            return []
        fids = [r["fact_id"] for r in rows]
        ages = np.array([max(t_now - r["written_t"], 0.0) for r in rows])
        lw = kernel.lognormal_logw(ages, S=self.S or config.KERNEL_S_DEFAULT)
        order = np.argsort(-lw)[:n]
        return [("fact", fids[i]) for i in order]

    def _pool_emo(self, t_now, n=30):
        """情绪路候选（**情节层**）：|效价|>=0.3 的经历按时间核取前 n 条。

        情节流才是带情绪标签的层 [DESIGN §2]；事实表按值去重后一条住院只剩
        一行，情绪召回会在那层失真。返 ("obs", obs_id)。
        """
        rows = self.store.con.execute(
            "SELECT obs_id, t, valence FROM obs "
            "WHERE valence IS NOT NULL AND ABS(valence)>=0.3 "
            "ORDER BY t DESC LIMIT 400").fetchall()
        if not rows:
            return []
        ages = np.array([max((t_now or 0.0) - r["t"], 0.0) for r in rows])
        lw = kernel.lognormal_logw(ages, S=self.S or config.KERNEL_S_DEFAULT)
        order = np.argsort(-lw)[:n]
        return [("obs", rows[i]["obs_id"]) for i in order]

    def _row_time(self, kind, rid):
        if kind == "fact":
            return self.store.fact_row(rid)["written_t"]
        return self.store.obs_row(rid)["t"]

    def _row_valence(self, kind, rid):
        if kind == "fact":
            return self._fact_valence(rid)
        obs = self.store.obs_row(rid)
        if obs["valence"] is not None:
            return float(obs["valence"])
        if obs["payload"]:
            from .extract import Extractor
            return Extractor.valence(obs["payload"])
        return 0.0

    def recall(self, query_text=None, query_vec=None, mood=0.0, t_now=None,
               k_gate=None, seed_ids=None):
        """三路召回 + 硬门槛（切在混合分上）[AUDIT §8 三步 + DESIGN §3.2 三路]。

        三条**平行**的召回路各自往候选池里放东西：
            语义路  fact/mention 向量 top-20   「你问明确的事」
            时间路  时间核 top-30（事实层）     「上次」「那天」
            情绪路  高效价经历 top-30（情节层）  「心情差想起糟心事」
        候选池内 z 标准化混合打分，硬门槛切前 k_gate 条。
        返回 [(kind, row_id, score)]；抽样是反应层的事。
        """
        k_gate = k_gate or config.GATE_K
        if query_vec is None and query_text is not None and self.embed_fn is not None:
            query_vec = self.embed_fn(query_text)
        # 第一层：三路并集召回（宽）
        sim_map = {}
        if query_vec is not None:
            for kind in ("fact", "mention"):
                for rid, s in self.semantic_topk(query_vec, kind=kind, k=20):
                    sim_map[(kind, rid)] = s
        cand = sorted(sim_map.keys(), key=lambda k: -sim_map[k])
        cand += self._pool_recent(t_now)
        cand += self._pool_emo(t_now)
        if seed_ids:
            cand += [("fact", f) for f in seed_ids]
        cand = list(dict.fromkeys(cand))
        if not cand:
            return []
        # 第二/三层：z 标准化混合分，硬门槛切前 k_gate 条（先筛后挑 [AUDIT §5]）。
        # 门槛切在混合分上：情绪路是平行召回路 [DESIGN §3.2]，必须能往筐里放
        # 东西；若先按 sim 砍到 k_gate 再混合，它就退化成筐内重排项（实测失灵）。
        sims = np.array([sim_map.get(c, 0.0) for c in cand])
        tws = np.array([float(kernel.weight(
            max((t_now or self._row_time(*c)) - self._row_time(*c), 0.0),
            S=self.S, sigma=self.sigma)) for c in cand])
        emos = np.array([1.0 + config.EMO_BIAS * mood * self._row_valence(*c)
                         for c in cand])
        score, _ = hybrid_score(sims, tws, emos)
        order = np.argsort(-score)
        return [(cand[i][0], cand[i][1], float(score[i])) for i in order[:k_gate]]

    def _fact_valence(self, fact_id):
        """事实的效价：优先用写入时评价出的 obs.valence，缺了退回小词典。缓存。"""
        cache = getattr(self, "_val_cache", None)
        if cache is None:
            cache = self._val_cache = {}
        if fact_id not in cache:
            row = self.store.fact_row(fact_id)
            val = 0.0
            if row is not None:
                obs = self.store.obs_row(row["src_obs"])
                if obs is not None:
                    if obs["valence"] is not None:
                        val = float(obs["valence"])
                    elif obs["payload"]:
                        from .extract import Extractor
                        val = Extractor.valence(obs["payload"])
            cache[fact_id] = val
        return cache[fact_id]

    # ------------------------------------------------ 过期降级判定
    def answer_or_degrade(self, entity, predicate, asof=None):
        ans = self.ask_fact(entity, predicate, asof=asof)
        return ans
