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

from .. import config
from . import kernel
from .scoring import hybrid_score


class Answer:
    """一次事实问答的完整出处。is_stale/is_guess 决定措辞，value 是内容。"""

    def __init__(self, values, rel=None, sources=None, stale=False, abstain=False,
                 degraded=False, fact_ids=None, retracted=False):
        self.values = values            # list[str]
        self.rel = rel                  # 来源分级（最高可信的那个）
        self.sources = sources or []    # 最早来源（谁最早说的）
        self.stale = stale              # 全部支撑已失效
        self.abstain = abstain
        self.degraded = degraded
        self.fact_ids = fact_ids or []  # 支撑本回答的事实行（动态修正锚定用）
        self.retracted = retracted      # 现行值为空、只有本人改口撤回的行
                                        # （「你之前说过X，后来你自己改口了」）

    def phrase(self, directness=None):
        """把 (值, 来源) 说成一句人话。

        `directness` 来自此刻的情绪（PAD 的 D 维，见 disposition）：D 高
        就敢下断言，D 低就先带上「我记得」这类铺垫再说话。

        为什么这根线要接在这里：之前 tone_block 会把「拿不准的要先说
        『我记得』」写进提示词，但快路径（点查命中、不调 LLM）根本读不到
        提示词——它直接走这个函数。于是「没底气」这件事在快路径上完全
        不存在，实测三档 PAD 下 `dh.ask` 的措辞一字不差。

        注意 stale 分支不参与软化：它本来就是「我记得以前是…现在不确定」，
        再加一层「我记得」会说成两道。
        """
        if self.abstain or not self.values:
            return config.ABSTAIN_PHRASE
        vals = "和".join(self.values)
        if self.retracted:
            # 本人改口 ≠ 记错：不是「我不确定」，是「我记得你的更正」——
            # 拟人细节在这里，失忆式跳过反而吓人（2026-11 改口协议审查 C5）
            return config.RETRACTED_PHRASE.format(old=vals)
        if self.stale:
            return config.EXPIRED_PHRASE.format(old=vals)
        # D 低 = 没把握：先垫一句再给内容。D 高不加（没有可加的余地：
        # 来源分级那套前缀本身就是「有多确定」的表达，删掉等于放弃幻觉治理）
        soft = (directness is not None
                and directness < config.PHRASE_SOFT_DIRECTNESS)
        if self.rel == 0:
            return ("我好像看到是%s" if soft else "我看到是%s") % vals
        if self.rel == 1:
            return ("我记得你告诉过我，是%s" if soft
                    else "你告诉过我，是%s") % vals
        if self.rel == 2 and self.sources:
            return ("好像是听%s说，是%s" if soft
                    else "听%s说，是%s") % (self.sources[0], vals)
        if self.rel == 3:
            # 已经是「猜」了，不叠词
            return "我猜是%s" % vals
        return ("我记得是%s" if soft else "是%s") % vals


class Retriever:
    def __init__(self, store, embed_fn=None, S=None, sigma=None, graph=None):
        self.store = store
        self.embed_fn = embed_fn
        self.S = S
        self.sigma = sigma
        # 图扩展用。DigitalHuman 建完图会回填（graph 持有同一个 store）；
        # 为 None 时 recall 退化成纯三路召回，行为与改动前完全一致
        self.graph = graph

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
        # 只剩本人改口撤回的行：不是弃权，是「记着你的更正」（近窗内）
        gone = self.store.retracted_values(
            entity, predicate,
            since=(asof if asof is not None else self._now())
                  - config.RETRACTED_CONTEXT_DAYS * 86400.0)
        if gone:
            return Answer([g["value"] for g in gone], rel=None, degraded=True,
                          retracted=True)
        return Answer([], abstain=True)

    def _now(self):
        import time
        return time.time()

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
        """语义 top-k：优先走 Chroma 索引，不可用就退回 numpy 暴力扫。

        两条路的返回值格式完全一致（[(row_id, cosine)] 降序），调用方
        不需要知道自己走的是哪条。索引挂了就退化成暴力扫——语义路变慢，
        但不会瞎，也不会抛异常把整轮对话拖崩。
        """
        if self.embed_fn is None or not callable(self.embed_fn):
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn <= 1e-9:
            # 全零查询向量（hash 嵌入遇到空串/纯符号时会出现）。
            # 归一化后 = 0 维扩散，所有 sim 变 inf/nan，必须早退
            return []
        # ---- 快路：Chroma（亚毫秒）。pending 里的新向量还没进索引，
        # 所以这条路要给它们留条后路——见下面 _merge_pending
        ix = self.store.vecindex() if config.VEC_INDEX_ON else None
        if ix is not None and ix.model == self._model_name() \
                and self.store.vec_count(kind, ix.model) >= config.VEC_INDEX_MIN_ROWS:
            hits = ix.query(kind, q, k=k if mask_ids is None else max(k * 4, 32))
            if hits:
                hits = self._merge_pending(kind, q, qn, hits, k)
                if mask_ids is not None:
                    allow = {int(i) for i in mask_ids}
                    hits = [(i, s) for i, s in hits if i in allow]
                return hits[:k]
        # ---- 慢路：暴力扫（兜底）
        ids, mat = self._mat(kind)
        if len(ids) == 0:
            return []
        if q.size != mat.shape[1]:
            return []
        # Accelerate/BLAS 的 float32 路径会对内部 denormal 置 FE_DIVBYZERO，
        # 虽然结果完全正确（实测 sims 全 finite），但会往 stderr 喷警告。
        # 只在这个 matmul 上关掉浮点信号，不动全局 seterr
        with np.errstate(all="ignore"):
            sims = mat @ (q / qn)
        if mask_ids is not None:
            allow = np.zeros(len(ids), dtype=bool)
            pos = {int(i): j for j, i in enumerate(ids)}
            for i in mask_ids:
                if int(i) in pos:
                    allow[pos[int(i)]] = True
            sims = np.where(allow, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [(int(ids[i]), float(sims[i])) for i in order if np.isfinite(sims[i])]

    def _merge_pending(self, kind, q, qn, hits, k):
        """把还没进索引的新向量用 numpy 补扫，并进 Chroma 结果。

        为什么必须有这一步：写入路径为了不掉 4.1 ms/条的索引开销，
        只把新向量记进 pending。如果检索只看 Chroma，**刚说过的话
        立刻就想不起来**——这是最刺眼的 bug（先问后答的场景 100% 撞上）。
        pending 一般只有几十条，补扫成本可忽略。
        """
        pend = self.store.pending(kind)
        if not pend:
            return hits
        want = self._model_name()
        rows = [(r[0], r[1]) for r in pend if r[2] == want]
        if not rows:
            return hits
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        mat = np.vstack([np.asarray(r[1], dtype=np.float32) for r in rows])
        if mat.shape[1] != q.size:
            return hits
        with np.errstate(all="ignore"):
            sims = mat @ (q / qn)
        best = {}
        for i, s in hits:
            best[int(i)] = max(best.get(int(i), -2.0), float(s))
        for j in np.argsort(-sims):
            fid = int(ids[j])
            s = float(sims[j])
            if s > best.get(fid, -2.0):
                best[fid] = s
        out = sorted(best.items(), key=lambda kv: -kv[1])
        return out[:max(k, len(hits))]

    def _model_name(self):
        """本检索器算出来的向量属于哪个模型——决定用哪个 collection。"""
        return getattr(self.embed_fn, "model_name", None) or self.store.dominant_model() \
            or "hash256"

    def _mat(self, kind):
        """向量矩阵缓存：行数没变就复用。

        先用便宜的 COUNT 判有效性，再决定要不要真去读 BLOB。
        原来的写法每次都先 `load_matrix`（一次 SQL + 上千次 frombuffer
        拼矩阵）再跟缓存比行数——缓存形同虚设，实测每轮白扔 3.5 ms。
        """
        if getattr(self, "_mat_cache", None) is None:
            self._mat_cache = {}
        model = self._model_name()
        n = self.store.vec_count(kind, model)
        cached = self._mat_cache.get(kind)
        if cached is not None and cached[0] == n and cached[2] == model:
            return cached[1], cached[3]
        ids, mat = self.store.load_matrix(kind, model=model)
        self._mat_cache[kind] = (len(ids), ids, model, mat)
        return ids, mat

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

    def _pool_emo(self, t_now, n=30, mood=0.0, n_trait=None):
        """情绪路候选（**情节层**）：带效价的经历，按「时间核 × 情绪一致」排。

        情节流才是带情绪标签的层 [DESIGN §2]；事实表按值去重后一条住院只剩
        一行，情绪召回会在那层失真。返 ("obs", obs_id)。

        **mood 必须在这里生效，不能只在后面的 hybrid_score 里加成权重。**
        实测过：mood 只进 0.2 权重时，候选池的**成员**不变（时间路固定取
        前 30 条），情绪只能在那 30 条内部重排，而 spoken 取 top3，所以
        四种情绪下说出来的东西一模一样。要让「心情差就想起糟心事」真的
        发生，情绪必须决定**谁能进池**。

        做法：一条记忆的「情绪契合度」= mood * valence（正性情绪配正性
        经历、负性配负性），折成乘法增益 exp(mood*valence*EMO_POOL_GAIN)。
        柔和指数而非线性相加，是因为线性会把池直接筛成同号的一半，太硬。

        编码强度（obs.salience，右脑 gate 落点）在此消费两根杠杆（以
        中性点位 EMO_SAL_BASE 为中心，老行/中性行杠杆恒为 0）：
            入场增益  lw += EMO_SAL_GAIN·(sal-BASE)——强情绪情节更容易想起
            遗忘减速  age_eff = age·exp(-SAL_AGE_SCALE·(sal-BASE))——强情绪
                      情节老化更慢（闪光灯记忆效应）

        性格杠杆（用户提案「存取对记忆加权」的性格半边，取侧）：
            n_trait = 特质 N（神经质，绝对值，成长层供数）。负性情节的
            有效年龄 ×exp(−GROWTH_N_AGE_SCALE·N·max(−valence,0))——敏感的
            人忘不掉伤心事、皮实的人翻篇更快；对正效价行恒为 1（皮实不
            对好事失忆）。None = 调用方没有成长层（裸测试），不加权。
        """
        rows = self.store.con.execute(
            "SELECT obs_id, t, valence, salience FROM obs "
            "WHERE valence IS NOT NULL AND ABS(valence)>=0.3 "
            "ORDER BY t DESC LIMIT 400").fetchall()
        if not rows:
            return []
        sal = np.array([config.EMO_SAL_BASE if r["salience"] is None
                        else float(r["salience"]) for r in rows])
        vals = np.array([float(r["valence"] or 0.0) for r in rows])
        ages = np.array([max((t_now or 0.0) - r["t"], 0.0) for r in rows])
        ages = ages * np.exp(-config.SAL_AGE_SCALE * (sal - config.EMO_SAL_BASE))
        if n_trait:
            hurt = np.clip(-vals, 0.0, 1.0)       # 只弯负效价记忆的曲线
            ages = ages * np.exp(-config.GROWTH_N_AGE_SCALE
                                 * float(n_trait) * hurt)
        lw = kernel.lognormal_logw(ages, S=self.S or config.KERNEL_S_DEFAULT)
        lw = lw + config.EMO_SAL_GAIN * (sal - config.EMO_SAL_BASE)
        if mood:
            lw = lw + mood * vals * config.EMO_POOL_GAIN
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
            from ..extract import Extractor
            return Extractor.valence(obs["payload"])
        return 0.0

    def related(self, seed_ids, k=3, exclude_entities=(), hops=None, ratio=None):
        """从若干事实出发，在激活图上找**别的实体**的相关记忆。

        这是图扩展在对话里真正有用的形态，跟"给候选池加分"是两件事：

        - 加分：扩展开的邻居去跟语义直中抢 k_gate 的位子。实测（lib5，
          k=8）会把直中从 40/40 挤到 19/40，得不偿失——所以 recall 里
          那条路只做补位。
        - 另开一节：问题点名老王时，把「老王老婆的病情」「同事件里别人
          的说法」单独列出来。这些记忆语义路一条都取不到（措辞完全不同），
          而它们正是"顺着话头聊下去"的原料。

        返回 [(fact_id, activation)]，已排除种子自身与 exclude_entities 里的
        实体，同一 (实体,谓词) 只留激活最高的一条。
        """
        if self.graph is None or not seed_ids:
            return []
        hops = config.RECALL_GRAPH_HOPS if hops is None else hops
        ratio = config.RECALL_GRAPH_RATIO if ratio is None else ratio
        seeds = {int(f): 1.0 for f in seed_ids}
        try:
            act = self.graph.diffuse(seeds, hops=hops)
        except Exception:
            return []
        for f in list(seeds):
            act.pop(f, None)
        if not act:
            return []
        peak = max(act.values())
        if peak <= 0:
            return []
        floor = peak * ratio
        skip = set(exclude_entities or ())
        best = {}
        for fid, a in act.items():
            if a < floor:
                continue
            row = self.store.fact_row(int(fid))
            if row is None or row["status"] == "retracted":
                continue
            if row["entity"] in skip:
                continue
            key = (row["entity"], row["predicate"])
            if key not in best or a > best[key][1]:
                best[key] = (int(fid), float(a))
        out = sorted(best.values(), key=lambda kv: -kv[1])
        return out[:k]

    def _expand_on_graph(self, sim_map):
        """语义 top-N 当种子在激活图上扩散，邻居折成「折扣继承」的相似度。

        三处口径必须说清，否则会写出一个分数尺度完全错的东西：

        1. **不能用激活值本身当相似度**。实测：种子节点激活 ~1.0，它的邻居
           只有 ~0.037（`A/√deg` 归一化 + 3 跳摊薄）。直接把 `a/max` 当余弦，
           邻居永远进不了候选池。正确做法是**折扣继承**：邻居的相关度 =
           种子自己的余弦 × 路径折扣。检索相关度是种子的相关度传下去的，
           不是图自己算出来的。
        2. **归一化基准要排掉种子**。扩散结果的 max 永远是种子自己（≈1.0），
           拿它当分母会把所有邻居压到 0.04 以下——这正是第一版写了却等于
           没写的原因（实测 added=0）。
        3. **图扩展只填坑，不抢位**。这一点有实测撑腰（lib5，1200 节点，
           事件成员 12 条）：让扩展开的候选与语义直中同池竞争，k=8 时
           事件命中 5→11，但**语义直中从 40/40 掉到 19/40**——扩展开的
           邻居余弦往往比直中更高（种子余弦×0.75 常常压过真正的第二名）。
           所以这里把扩展候选的分数压到「所有直中之下的最高一档」，
           它只在语义路凑不满 k_gate 时补位。图是补漏，不是替代。
        """
        if not config.RECALL_GRAPH_ON or self.graph is None or not sim_map:
            return sim_map, set()
        seeds = [(k, v) for k, v in sim_map.items() if k[0] == "fact"]
        if not seeds:
            return sim_map, set()
        seeds.sort(key=lambda kv: -kv[1])
        top_seeds = seeds[:config.RECALL_GRAPH_SEEDS]
        # 直中的地板分：扩展候选一律压在它下面，保证一个直中都不会被挤掉
        direct_floor = min(sim_map.values())
        out = dict(sim_map)
        for (_, sid), sim_s in top_seeds:
            sid = int(sid)
            try:
                act = self.graph.diffuse({sid: 1.0}, hops=config.RECALL_GRAPH_HOPS)
            except Exception:
                return out, set()       # 图坏了不能拖垮检索
            act.pop(sid, None)          # 归一化基准排掉种子自己
            if not act:
                continue
            peak = max(act.values())
            if peak <= 0:
                continue
            floor = peak * config.RECALL_GRAPH_RATIO
            for fid, a in act.items():
                if a < floor:
                    continue
                key = ("fact", int(fid))
                if key in sim_map:
                    continue        # 直中已有，绝不动它的分
                w = direct_floor * config.RECALL_GRAPH_W * (a / peak) / max(peak, 1e-9)
                w = min(w, direct_floor)
                if w > out.get(key, 0.0):
                    out[key] = w
        added = {k for k in out if k not in sim_map}
        return out, added

    def recall(self, query_text=None, query_vec=None, mood=0.0, t_now=None,
               k_gate=None, seed_ids=None, n_trait=None):
        """三路召回 + 硬门槛（切在混合分上）[AUDIT §8 三步 + DESIGN §3.2 三路]。

        三条**平行**的召回路各自往候选池里放东西：
            语义路  fact/mention 向量 top-20   「你问明确的事」
            时间路  时间核 top-30（事实层）     「上次」「那天」
            情绪路  高效价经历 top-30（情节层）  「心情差想起糟心事」
        候选池内 z 标准化混合打分，硬门槛切前 k_gate 条。
        n_trait：特质 N（取侧性格杠杆，见 _pool_emo）；None 不加权。
        返回 [(kind, row_id, score)]；抽样是反应层的事。
        """
        k_gate = k_gate or config.GATE_K
        if query_vec is None and query_text is not None and self.embed_fn is not None:
            query_vec = self.embed_fn(query_text)
        # 第一层：三路并集召回（宽）
        # 语义路扫 fact（抽取出的三元组）+ obs（原话）。
        # obs 这条不能少：正则抽不出的句子（「老王搬海淀去了」）在 fact 层
        # 根本不存在，只有原话向量能把它们捞回来。两条一起扫，谁的余弦高
        # 谁进池——但不是同一个 id 空间，所以 (kind, rid) 的 kind 必须带着走。
        sim_map = {}
        if query_vec is not None:
            for kind in ("fact", "obs"):
                for rid, s in self.semantic_topk(query_vec, kind=kind, k=20):
                    sim_map[(kind, rid)] = s
        # 第一点五层：图上扩展（RAG 的标准形状）。
        # 语义路只认"字面/语义近"，认不出"这条记忆跟那条记忆在同一个事件里"
        # ——那个信息只存在图里（link 边=同事件、share 边=同一个人）。
        # 做法：语义路 top-N（默认 5）当种子，扩散 2 跳，激活值折算成相似度
        # 后与原有 sim **取 max**（不相加：相加会把两条路的口径混成一个假的分数）。
        sim_map, graph_added = self._expand_on_graph(sim_map)
        cand = sorted(sim_map.keys(), key=lambda k: -sim_map[k])
        cand += self._pool_recent(t_now)
        cand += self._pool_emo(t_now, mood=mood, n_trait=n_trait)
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
        # sim_map 要带进配额层：fact 想占位得证明自己跟问题有关（见 _quota_pick）。
        picked = self._quota_pick(cand, score, order, k_gate, graph_added, sim_map)
        return [(cand[i][0], cand[i][1], float(score[i])) for i in picked]

    def _quota_pick(self, cand, score, order, k_gate, graph_added, sim_map=None):
        """按配额切：图扩展来的候补最多占 RECALL_GRAPH_QUOTA，
        原话（obs）最多占 RECALL_OBS_QUOTA，其余留给结构化事实。

        为什么必须限配额：实测不限制时，图邻域会把语义直中挤掉
        （k=8 下直中从 40/40 掉到 8/40）；原话同理——obs 条数是 fact 的
        约 80 倍（1773 条原话 → 22 条事实），语义 top-20 里几乎全是原话，
        z 标准化后 sim 项占一半权重，原话天然把事实挤出前 k 条。
        实测 300 条闲聊 + 10 条事实：「我姐在哪」8/8 全是 obs。
        图是补漏、原话也是补漏（正则抽不出的句子只能靠它），
        两者都不该抢主路——精确的结构化事实才是答案。

        但配额只在**有人来填**时才生效：如果池里压根没有 fact，
        最后一遍补位会把剩下的空名额还给 obs，不会白留空位。

        sim_map 传进来是为了区分"跟问题有关的 fact"和"碰巧在池里的 fact"。
        没有这道判断时 obs 配额是个死数（k_gate 的一半），于是不管那几个
        fact 相不相关都会占掉另外一半：实测问「张姨住哪」时，老王的三条事
        实余弦只有 0.36/0.40/0.46，却因为配额被硬塞进 8 条答案里，把真正
        相关的原话（0.78）挤了出去。反过来「老王电话多少」时电话事实余弦
        0.88，本来就该占位。两者唯一的差别就是余弦，所以门槛切在余弦上：
        实测相关 fact 0.65-0.88、不相关 0.25-0.46，0.5 正好在中间。
        """
        min_cos = config.RECALL_FACT_MIN_COS

        def _qualified(i):
            """这条候选有没有资格按自己的分数排位。

            fact 在语义路里被算过分、但分不够的，说明它跟这个问题没关系，
            排到队尾——仍然可能补位进来（总比空着好），只是不能抢在相关的
            原话前面。没被语义路算过分的 fact（时间路/情绪路捞出来的）不在
            此列：它们本来就不靠余弦进来，让它们按时间核/效价的分数排。
            """
            key = cand[i]
            if key[0] != "fact" or not sim_map or key not in sim_map:
                return True
            # 图扩展进来的 fact 不算：它的 sim_map 值是扩散折扣后的激活值
            # （RECALL_GRAPH_W 折算，天然低于门槛），拿它跟直中余弦比不公平。
            # 这类候选本来就另有一套配额（RECALL_GRAPH_QUOTA）。
            if graph_added and key in graph_added:
                return True
            return sim_map[key] >= min_cos

        n_elig_fact = sum(1 for i in range(len(cand))
                          if cand[i][0] == "fact" and _qualified(i))
        # obs 的可用名额跟着"够格的事实"走：事实够格占多少，obs 就让多少；
        # 池里没有够格的事实时可占满（RECALL_OBS_QUOTA 当地板，不是死上限）。
        obs_quota = min(k_gate, max(k_gate - n_elig_fact,
                                    int(round(k_gate * config.RECALL_OBS_QUOTA))))
        order = [i for i in order if _qualified(i)] + \
                [i for i in order if not _qualified(i)]
        graph_quota = int(round(k_gate * config.RECALL_GRAPH_QUOTA))
        picked, extra, n_obs = [], 0, 0
        for i in order:
            if len(picked) >= k_gate:
                break
            key = cand[i]
            if graph_added and key in graph_added and extra >= graph_quota:
                continue
            if key[0] == "obs" and n_obs >= obs_quota:
                continue                                    # 先让位于 fact
            if graph_added and key in graph_added:
                extra += 1
            if key[0] == "obs":
                n_obs += 1
            picked.append(i)
        # 补位：配额卡掉的空名额还给剩下的候选（池里没 fact 时 obs 该占满）
        if len(picked) < k_gate:
            taken = set(picked)
            for i in order:
                if len(picked) >= k_gate:
                    break
                if i not in taken:
                    picked.append(i)
        return picked

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
                        from ..extract import Extractor
                        val = Extractor.valence(obs["payload"])
            cache[fact_id] = val
        return cache[fact_id]

    # ------------------------------------------------ 过期降级判定
    def answer_or_degrade(self, entity, predicate, asof=None):
        ans = self.ask_fact(entity, predicate, asof=asof)
        return ans
