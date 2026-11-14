# -*- coding: utf-8 -*-
"""存储层：一个 SQLite 文件 + 一块 numpy 矩阵。

五个区不是「短期/中期/长期」，是按「能不能改、谁改、改多快」划分 [STORE §1]：
    obs        原始观测   物理禁止改（触发器锁死）
    fact       事实       一行 = 一个值的一段有效期（双时态），只加新行不删旧行
    event/mention/slot    同一件事的牌子 + 说法 + 格子
    self_narr  自我叙事   极少，几乎不改
    vec        向量       BLOB，检索时整块读进内存暴力扫
    calib      遗忘常数   会话级一个数，只插不改

性能预期（本机实测口径 [STORE §5]）：点查微秒级、写几十万条/秒、
20 万条向量暴力扫 5-17 ms —— 所以不引入任何向量库/图库/缓存。
"""
import os
import sqlite3
import struct
import threading

import numpy as np

from .. import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS obs(
  obs_id   INTEGER PRIMARY KEY,
  t        REAL NOT NULL,
  entity   TEXT,
  kind     TEXT NOT NULL,      -- perceive / told / infer / self
  channel  TEXT NOT NULL,      -- camera / mic / text / api
  source   TEXT,               -- 谁说的 / 哪个摄像头（只是属性，不是图节点 [GRAPH §10.1]）
  rel      INTEGER NOT NULL,   -- 来源分级 L0..L4
  seg_bnd  REAL NOT NULL DEFAULT 0,   -- 是否为一段对话的开头（>30min 空隙）
  valence  REAL,               -- 写入时评价出的效价（情节记忆的情绪标签 [DESIGN §2]）
  payload  TEXT,
  salience REAL NOT NULL DEFAULT 0.15 -- 情绪编码强度 gate∈[0,1]，中性=EMO_SAL_BASE
);
CREATE TRIGGER IF NOT EXISTS obs_no_update BEFORE UPDATE ON obs
BEGIN SELECT RAISE(ABORT, 'obs is append-only'); END;
CREATE TRIGGER IF NOT EXISTS obs_no_delete BEFORE DELETE ON obs
BEGIN SELECT RAISE(ABORT, 'obs is append-only'); END;
CREATE INDEX IF NOT EXISTS ix_obs_t ON obs(t);

CREATE TABLE IF NOT EXISTS pred(
  predicate TEXT PRIMARY KEY,
  multi     INTEGER NOT NULL DEFAULT 0,   -- 1=多值并存, 0=单值互斥
  unit      TEXT
);

CREATE TABLE IF NOT EXISTS fact(
  fact_id   INTEGER PRIMARY KEY,
  entity    TEXT NOT NULL,
  predicate TEXT NOT NULL,
  value     TEXT NOT NULL,
  valid_from REAL NOT NULL,
  valid_to   REAL,              -- NULL = 仍然有效
  status    TEXT NOT NULL,      -- active / superseded / retracted
  src_obs   INTEGER NOT NULL REFERENCES obs(obs_id),
  rel       INTEGER NOT NULL,
  conf      REAL NOT NULL,
  written_t REAL NOT NULL,
  confirmed_t REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fact_ep ON fact(entity, predicate, status);
CREATE INDEX IF NOT EXISTS ix_fact_val ON fact(entity, predicate, value);

CREATE TABLE IF NOT EXISTS event(
  event_id INTEGER PRIMARY KEY,
  label    TEXT NOT NULL,
  key      TEXT,               -- 主体|谓词（自动聚类的键）
  t_start  REAL, t_end REAL,
  created_t REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mention(
  mention_id INTEGER PRIMARY KEY,
  event_id   INTEGER NOT NULL REFERENCES event(event_id),
  obs_id     INTEGER NOT NULL REFERENCES obs(obs_id),
  text       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS slot(
  mention_id INTEGER NOT NULL REFERENCES mention(mention_id),
  name  TEXT NOT NULL,
  value TEXT NOT NULL,
  PRIMARY KEY(mention_id, name)
);

CREATE TABLE IF NOT EXISTS vec(
  kind   TEXT NOT NULL,     -- fact / mention / obs
  row_id INTEGER NOT NULL,
  dim    INTEGER NOT NULL,
  v      BLOB NOT NULL,
  model  TEXT NOT NULL DEFAULT 'hash256',   -- 是哪个模型算的；不同模型余弦不可比
  PRIMARY KEY(kind, row_id)
);

CREATE TABLE IF NOT EXISTS self_narr(
  id INTEGER PRIMARY KEY,
  claim TEXT NOT NULL,
  weight REAL NOT NULL,
  updated_t REAL NOT NULL
);

-- 标定表：只插不改，读的时候取 fitted_t 最新的一行 [STORE §6]
CREATE TABLE IF NOT EXISTS calib(
  id INTEGER PRIMARY KEY,
  scope TEXT NOT NULL,
  key   TEXT NOT NULL,
  k     TEXT NOT NULL,
  v     REAL NOT NULL,
  note  TEXT,
  fitted_t REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_calib ON calib(scope, key, k, fitted_t);

-- 别名词典：LLM 抽取发现的新实体回写到这里，正则快路下次直接命中
CREATE TABLE IF NOT EXISTS alias(
  name  TEXT PRIMARY KEY,
  canon TEXT NOT NULL
);

-- 值别名：写入侧等价裁决（dh/memory/equiv.py）发现的「同一事实的另一种说法」。
-- 记的是 (实体,谓词,表面形式)->规范值，下次同表面形式直接命中，不再算余弦。
-- 只增不改；事实层的值字段永远是规范值，表面形式只活在这里和 obs 原文里。
CREATE TABLE IF NOT EXISTS value_alias(
  entity    TEXT NOT NULL,
  predicate TEXT NOT NULL,
  surface   TEXT NOT NULL,
  canonical TEXT NOT NULL,
  learned_t REAL NOT NULL,
  PRIMARY KEY(entity, predicate, surface)
);

-- 谓词问法线索：问句路由器（dh/memory/qroute.py）的 L1 教材。
-- source='seed' 是出厂种子（bootstrap），source='llm' 是 L2 解析成功后
-- 回填的自学问法——表自己生长，零人工。向量进 vec 表 kind='pred_hint'。
CREATE TABLE IF NOT EXISTS pred_hint(
  hint_id   INTEGER PRIMARY KEY,
  predicate TEXT NOT NULL,
  question  TEXT NOT NULL,
  source    TEXT NOT NULL DEFAULT 'llm',
  learned_t REAL NOT NULL,
  UNIQUE(predicate, question)
);

-- 元信息：属主（正在对话的那个人）等。属主是代词消解的锚——
-- 说话人句子里的「我」确定性地指属主，不靠 LLM 自觉
CREATE TABLE IF NOT EXISTS meta(
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- 激活图的持久部分 [GRAPH §6]；运行时读成 CSR，不碰 SQLite
CREATE TABLE IF NOT EXISTS node(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,          -- entity / fact / obs / event
  label TEXT NOT NULL,
  t REAL,
  salience REAL DEFAULT 0.5,
  ref_table TEXT,
  ref_id INTEGER
);
CREATE TABLE IF NOT EXISTS edge(
  a INTEGER NOT NULL,
  b INTEGER NOT NULL,
  kind TEXT NOT NULL,          -- link / share / sim
  w REAL NOT NULL,
  src TEXT,                    -- 来源只是边上的属性 [GRAPH §10.1]
  PRIMARY KEY(a, b, kind)
);

CREATE TABLE IF NOT EXISTS qa_log(
  qa_id INTEGER PRIMARY KEY,
  q TEXT NOT NULL,
  t REAL NOT NULL,
  outcome TEXT NOT NULL        -- ok / wrong / abstain
);
"""


class MemoryStore:
    """数字人的全部持久状态。一个文件，五个区。"""

    def __init__(self, path=":memory:"):
        # 记住库路径：视图类文件（data/persona.md）要按「这个库是不是真的
        # 落盘库」决定写不写，内存库不能去覆盖磁盘上的真值视图
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # check_same_thread=False：网页服务是多线程的，访问用 self.lock 串行化
        #（单用户聊天页，粗粒度 RLock 足够）
        self.lock = threading.RLock()
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row      # 允许按列名取，避免位置索引踩坑
        self.con.execute("PRAGMA journal_mode=WAL")
        # WAL 默认攒 4MB 才自动合并——进程被强杀时未合并的帧可能丢。
        # 降到 64 页（256KB）+ commit 后主动 checkpoint，重启零丢失
        self.con.execute("PRAGMA wal_autocheckpoint=64")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript(SCHEMA)
        self._migrate()
        self._fts = True
        try:
            self.con.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS obs_fts USING fts5("
                "payload, content='obs', content_rowid='obs_id')")
        except sqlite3.OperationalError:
            self._fts = False        # 没有 FTS5 就靠全表扫，不挡功能
        # 向量索引（Chroma）。只对落盘库启用——内存库每次都是新的，
        # 建索引的固定开销（HNSW 落盘 + 加载）比暴力扫还贵
        self._vecindex = None
        self._dominant_model = None
        self._vec_rows = {}          # kind -> 行数缓存（写一次就失效）
        # kind -> {row_id: (vec, model)}，待入索引。用 dict 而不是 list：
        # 同一事实会被反复提及（write_fact 返回同一个 fact_id），list 会把
        # 同一行攒成几十份，1800 轮后 pending 有 1679 条但唯一行只有 22 行
        self._pending = {}

    # ------------------------------------------------ 迁移
    def _migrate(self):
        """老库补列。SQLite 没有 ADD COLUMN IF NOT EXISTS，只能先查 pragma。

        vec.model：嵌入模型标识。老库全按 hash256 记（那时候只有哈希嵌入），
        换模型后 load_matrix 会按 model 过滤，混着算余弦等于算噪声。
        obs.salience：情绪编码强度（[DESIGN §4.3] 情节门控）。老库全按中性
        点位 EMO_SAL_BASE=0.15 记——读出侧杠杆以它为中心，老行不受影响。
        """
        cols = {r["name"] for r in self.con.execute("PRAGMA table_info(vec)")}
        if "model" not in cols:
            self.con.execute(
                "ALTER TABLE vec ADD COLUMN model TEXT NOT NULL DEFAULT 'hash256'")
            self.con.commit()
        cols = {r["name"] for r in self.con.execute("PRAGMA table_info(obs)")}
        if "salience" not in cols:
            self.con.execute(
                "ALTER TABLE obs ADD COLUMN salience REAL NOT NULL DEFAULT 0.15")
            self.con.commit()

    # ------------------------------------------------ 观测
    def add_obs(self, t, kind, channel, payload, entity=None, source=None,
                rel=2, seg_bnd=0.0, valence=None, salience=None):
        """salience：情绪编码强度 ∈[0,1]（右脑 gate 落点）。

        不传时取中性点位 config.EMO_SAL_BASE（0.15）——所有杠杆以它为
        中心，不传即「中性行为与未接线前一致」。它不决定行存不存在——
        obs 追加锁死，一句经历永远留底；它决定这段情节「记得多深」：
        检索入场增益 + 遗忘减速（retrieve._pool_emo）、塑形权重
        （limbic.persona.shape）、游移回味偏好（core/offline）。"""
        cur = self.con.execute(
            "INSERT INTO obs(t,entity,kind,channel,source,rel,seg_bnd,valence,"
            "payload,salience) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (t, entity, kind, channel, source, int(rel), float(seg_bnd),
             None if valence is None else float(valence), payload,
             float(min(1.0, max(0.0, config.EMO_SAL_BASE if salience is None
                                else salience)))))
        oid = cur.lastrowid
        if self._fts and payload:
            self.con.execute("INSERT INTO obs_fts(rowid,payload) VALUES(?,?)", (oid, payload))
        return oid

    def obs_count(self):
        return self.con.execute("SELECT COUNT(*) FROM obs").fetchone()[0]

    # ------------------------------------------------ 谓词
    def declare_pred(self, predicate, multi=0, unit=None):
        self.con.execute(
            "INSERT INTO pred(predicate,multi,unit) VALUES(?,?,?) "
            "ON CONFLICT(predicate) DO NOTHING", (predicate, int(multi), unit))

    def is_multi(self, predicate):
        row = self.con.execute("SELECT multi FROM pred WHERE predicate=?",
                               (predicate,)).fetchone()
        return bool(row[0]) if row else False

    # ------------------------------------------------ 事实写入
    def write_fact(self, entity, predicate, value, t, obs_id, rel=2, conf=0.9,
                   force_new=False, revive_ok=False):
        """写一条事实。规则 [STORE §2 + 第六阶段传旧话处理]：

        单值谓词：写新的、关掉旧的（status=superseded，不删）。
        多值谓词：写新的、旧的留着（两行同时 active）。
        重复提到同一个值：
          - 该值仍 active      -> 只更新 confirmed_t（告警位，不进权重）
          - 该值已失效         -> 默认只更新旧行的 confirmed_t，**不复活**——
                                  对话里复述旧值十有八九是传旧话（世界 2.7% 的
                                  轮次；φ/back 分析：真实世界不会随便翻回旧值）。
                                  例外：本人亲口宣布变更（revive_ok，如「我换号了」）
                                  且文本带变更信号 -> 复活为新行。
        返回 (fact_id, action)，action ∈ new / confirm / confirm_old / revive。
        """
        multi = self.is_multi(predicate)
        revived = False
        same = self.con.execute(
            "SELECT fact_id,status FROM fact WHERE entity=? AND predicate=? AND value=? "
            "ORDER BY fact_id DESC LIMIT 1", (entity, predicate, value)).fetchone()
        if same is not None:
            fid, st = same
            if st == "active":
                # 还生效的值被再提：只动 confirmed_t（告警位），force_new 也不加行
                self.con.execute("UPDATE fact SET confirmed_t=? WHERE fact_id=?", (t, fid))
                return fid, "confirm"
            if force_new and revive_ok:
                revived = True                          # 本人宣布改回来 -> 落到插入
            else:
                # 传旧话：复述失效值不复活
                self.con.execute("UPDATE fact SET confirmed_t=? WHERE fact_id=?", (t, fid))
                return fid, "confirm_old"
        if not multi:
            self.con.execute(
                "UPDATE fact SET status='superseded', valid_to=? "
                "WHERE entity=? AND predicate=? AND status='active'", (t, entity, predicate))
        cur = self.con.execute(
            "INSERT INTO fact(entity,predicate,value,valid_from,valid_to,status,"
            "src_obs,rel,conf,written_t,confirmed_t) VALUES(?,?,?,?,NULL,'active',?,?,?,?,?)",
            (entity, predicate, value, t, obs_id, int(rel), float(conf), t, t))
        # revive 与 new 落同样的行，但动作名分开——「复活的旧值」和「全新值」
        # 在调用方（确认回复/诊断）不是一个语义（docstring 一直这么写，代码此前没兑现）
        return cur.lastrowid, "revive" if revived else "new"

    # ------------------------------------------------ 事实查询
    def active_values(self, entity, predicate, asof=None):
        """当前生效的值。多值谓词可能多行；单值谓词最多一行。"""
        cond = "AND valid_from<=?" if asof is not None else ""
        args = [entity, predicate] + ([asof] if asof is not None else [])
        return self.con.execute(
            "SELECT fact_id,value,valid_from,valid_to,rel,conf,src_obs,confirmed_t "
            "FROM fact WHERE entity=? AND predicate=? AND status='active' " + cond +
            " ORDER BY valid_from", args).fetchall()

    def value_at(self, entity, predicate, t):
        """时间旅行：t 那一刻是什么 [STORE §2 问3]。"""
        return self.con.execute(
            "SELECT fact_id,value,valid_from,valid_to,status,rel,conf,src_obs "
            "FROM fact WHERE entity=? AND predicate=? AND valid_from<=? "
            "AND (valid_to IS NULL OR valid_to>?) ORDER BY valid_from",
            (entity, predicate, t, t)).fetchall()

    def history(self, entity, predicate):
        return self.con.execute(
            "SELECT value,valid_from,COALESCE(valid_to,-1) AS valid_to,status,rel "
            "FROM fact WHERE entity=? AND predicate=? ORDER BY valid_from",
            (entity, predicate)).fetchall()

    def superseded_values(self, entity, predicate):
        return self.con.execute(
            "SELECT value,valid_from,valid_to FROM fact "
            "WHERE entity=? AND predicate=? AND status='superseded' "
            "ORDER BY valid_from DESC", (entity, predicate)).fetchall()

    def fact_row(self, fact_id):
        return self.con.execute("SELECT * FROM fact WHERE fact_id=?", (fact_id,)).fetchone()

    def obs_row(self, obs_id):
        return self.con.execute("SELECT * FROM obs WHERE obs_id=?", (obs_id,)).fetchone()

    def fact_count(self):
        return self.con.execute("SELECT COUNT(*) FROM fact").fetchone()[0]

    # ------------------------------------------------ 事件
    def open_event(self, label, t, key=None):
        cur = self.con.execute(
            "INSERT INTO event(label,key,t_start,t_end,created_t) VALUES(?,?,?,NULL,?)",
            (label, key, t, t))
        return cur.lastrowid

    def latest_open_event(self, key, t, window):
        """同 (主体|谓词) 在时间窗内的开放事件；没有就 None。"""
        row = self.con.execute(
            "SELECT event_id,t_start FROM event WHERE key=? AND t_end IS NULL "
            "ORDER BY event_id DESC LIMIT 1", (key,)).fetchone()
        if row and t - row[1] <= window:
            return row[0]
        return None

    def close_event(self, event_id, t_end):
        self.con.execute("UPDATE event SET t_end=? WHERE event_id=?", (t_end, event_id))

    def add_mention(self, event_id, obs_id, text):
        cur = self.con.execute(
            "INSERT INTO mention(event_id,obs_id,text) VALUES(?,?,?)",
            (event_id, obs_id, text))
        return cur.lastrowid

    def add_slot(self, mention_id, name, value):
        self.con.execute(
            "INSERT INTO slot(mention_id,name,value) VALUES(?,?,?) "
            "ON CONFLICT(mention_id,name) DO UPDATE SET value=excluded.value",
            (mention_id, name, value))

    def siblings(self, mention_id):
        """同一块牌子下的其他说法 [STORE §2 问5]。"""
        return self.con.execute(
            "SELECT m2.mention_id,m2.text FROM mention m1 "
            "JOIN mention m2 ON m2.event_id=m1.event_id AND m2.mention_id<>m1.mention_id "
            "WHERE m1.mention_id=?", (mention_id,)).fetchall()

    def event_slots(self, event_id):
        return self.con.execute(
            "SELECT name,COUNT(*) FROM slot WHERE mention_id IN "
            "(SELECT mention_id FROM mention WHERE event_id=?) GROUP BY name",
            (event_id,)).fetchall()

    # ------------------------------------------------ 向量
    def put_vec(self, kind, row_id, vec, model=None):
        """写一条向量。model 记来源——不同模型的余弦不可比，
        换模型时必须整表重算（load_matrix 会按 model 过滤）。"""
        v = np.asarray(vec, dtype=np.float32)
        model = model or "hash256"
        self.con.execute(
            "INSERT INTO vec(kind,row_id,dim,v,model) VALUES(?,?,?,?,?) "
            "ON CONFLICT(kind,row_id) DO UPDATE SET dim=excluded.dim,"
            "v=excluded.v, model=excluded.model",
            (kind, row_id, int(v.size), v.tobytes(), model))
        # **不在这里写 Chroma**。单条 upsert 实测 4.1 ms——摄取 1800 轮
        # 就是 7 秒的纯索引开销，而且这些写发生在对话锁里。
        # 索引是派生物：先只落 SQLite（真值），把新行记进 pending，
        # 检索时由 _flush_pending 成批送，或攒到阈值再送。
        self._vec_rows.pop(kind, None)      # 行数变了，缓存失效
        bucket = self._pending.setdefault(kind, {})
        bucket[int(row_id)] = (v, model)
        # 攒够一批就送。阈值以下留在 pending（检索时 numpy 补扫，
        # 保证"刚说过的话立刻能想起来"），超过就成批入 HNSW
        if len(bucket) >= config.VEC_INDEX_FLUSH_AT:
            self.flush_pending(kind, model=model)

    def load_matrix(self, kind, model=None):
        """整块读出成 (ids, matrix)。矩阵常驻内存，检索时暴力扫 [STORE §4]。

        model=None 时取该 kind 下**出现最多的那个模型**，只读那一批——
        混着读会把 256 维哈希和 512 维 bge 拼进一个矩阵（要么维度崩，
        要么维度正好相同但语义空间不同，余弦全是噪声）。
        """
        if model is None:
            row = self.con.execute(
                "SELECT model FROM vec WHERE kind=? GROUP BY model "
                "ORDER BY COUNT(*) DESC LIMIT 1", (kind,)).fetchone()
            if row is None:
                return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
            model = row[0]
        rows = self.con.execute(
            "SELECT row_id,dim,v FROM vec WHERE kind=? AND model=? ORDER BY row_id",
            (kind, model)).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        dim = rows[0][1]
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        mat = np.vstack([np.frombuffer(r[2], dtype=np.float32, count=dim) for r in rows])
        return ids, mat

    def vec_count(self, kind, model=None):
        """该 kind 有多少条向量。带缓存——检索每轮都要问，不能每次 COUNT(*)。"""
        model = model or self.dominant_model()
        key = (kind, model)
        if key in self._vec_rows:
            return self._vec_rows[key]
        row = self.con.execute(
            "SELECT COUNT(*) FROM vec WHERE kind=? AND model=?",
            (kind, model or "")).fetchone()
        n = int(row[0]) if row else 0
        self._vec_rows[key] = n
        return n

    def pending(self, kind):
        """还没进 Chroma 的增量 [(row_id, vec, model)]。检索时 numpy 补扫。"""
        return [(rid, v, m) for rid, (v, m) in self._pending.get(kind, {}).items()]

    def flush_pending(self, kind=None, model=None):
        """把 pending 成批送进索引。返回送出条数。"""
        ix = self.vecindex(model)
        if ix is None:
            return 0
        kinds = [kind] if kind else list(self._pending)
        n = 0
        for k in kinds:
            bucket = self._pending.get(k, {})
            rows = [(rid, v) for rid, (v, m) in bucket.items() if m == ix.model]
            if not rows:
                continue
            n += ix.upsert(k, rows)
            for rid, _v in rows:
                bucket.pop(rid, None)
        return n

    def vecindex(self, model=None):
        """当前嵌入模型对应的 Chroma 索引。内存库返回 None（走暴力扫）。"""
        if self.path == ":memory:":
            return None
        model = model or self.dominant_model()
        if model is None:
            return None
        if self._dominant_model is None:
            self._dominant_model = model
        ix = getattr(self, "_vecindex", None)
        if ix is not None and ix.model == model:
            return ix
        from .vecindex import VectorIndex, default_dir
        d = default_dir(self.path)
        if d is None:
            return None
        self._vecindex = VectorIndex(d, model=model)
        return self._vecindex

    def dominant_model(self):
        """库里主用哪个嵌入模型。一个库只用一种，混着算余弦等于算噪声。

        结果缓存：put_vec 每次都要问一遍，不缓存就是每条记忆一次 GROUP BY。
        换模型要重开进程（换模型本来就得整表重算）。
        """
        if self._dominant_model is None:
            row = self.con.execute(
                "SELECT model FROM vec GROUP BY model ORDER BY COUNT(*) DESC LIMIT 1"
            ).fetchone()
            if row:
                self._dominant_model = row[0]
        return self._dominant_model

    def sync_vecindex(self):
        """启动时对齐一次。数量不等就整目录重建索引。"""
        ix = self.vecindex()
        if ix is None:
            return {}
        self.flush_pending()
        return ix.sync(self)

    def vec_models(self, kind):
        """该 kind 下每种模型各有多少条——用来判断要不要重算。"""
        return {r[0]: r[1] for r in self.con.execute(
            "SELECT model,COUNT(*) FROM vec WHERE kind=? GROUP BY model",
            (kind,)).fetchall()}

    # ------------------------------------------------ 自我叙事
    def add_self_narr(self, claim, weight, t):
        cur = self.con.execute(
            "INSERT INTO self_narr(claim,weight,updated_t) VALUES(?,?,?)",
            (claim, weight, t))
        return cur.lastrowid

    def self_narrs(self):
        return self.con.execute(
            "SELECT id,claim,weight,updated_t FROM self_narr ORDER BY weight DESC").fetchall()

    # ------------------------------------------------ 别名词典
    def alias_learn(self, name, canon):
        self.con.execute(
            "INSERT INTO alias(name,canon) VALUES(?,?) "
            "ON CONFLICT(name) DO NOTHING", (name, canon))

    def alias_all(self):
        return self.con.execute("SELECT name,canon FROM alias").fetchall()

    # ------------------------------------------------ 值别名（写入侧等价裁决）
    def value_alias_learn(self, entity, predicate, surface, canonical, t=0.0):
        """登记「这个表面形式说的是那个规范值」。surface==canonical 不记。"""
        if surface == canonical or not surface or not canonical:
            return False
        self.con.execute(
            "INSERT INTO value_alias(entity,predicate,surface,canonical,learned_t) "
            "VALUES(?,?,?,?,?) ON CONFLICT(entity,predicate,surface) DO NOTHING",
            (entity, predicate, surface, canonical, float(t)))
        return True

    def value_alias_lookup(self, entity, predicate, surface):
        return self.con.execute(
            "SELECT canonical FROM value_alias "
            "WHERE entity=? AND predicate=? AND surface=?",
            (entity, predicate, surface)).fetchone()

    def confirm_fact(self, fact_id, t):
        """等价确认：只动 confirmed_t 告警位。不加行、不 supersede、不动
        valid_from——语义相同的复述不该在双时态历史里留下一次假变更。"""
        self.con.execute(
            "UPDATE fact SET confirmed_t=? WHERE fact_id=?", (float(t), fact_id))

    # ------------------------------------------------ 谓词问法线索（问句路由）
    def pred_hint_learn(self, predicate, question, t=0.0, source="llm"):
        """登记一条问法线索（幂等：同谓词同问法不重复）。返回 hint_id 或 None。"""
        cur = self.con.execute(
            "INSERT INTO pred_hint(predicate,question,source,learned_t) "
            "VALUES(?,?,?,?) ON CONFLICT(predicate,question) DO NOTHING",
            (predicate, question, source, float(t)))
        return cur.lastrowid if cur.rowcount else None

    def pred_hints_all(self):
        return self.con.execute(
            "SELECT hint_id,predicate,question,source,learned_t FROM pred_hint "
            "ORDER BY hint_id").fetchall()

    # ------------------------------------------------ 元信息（属主等）
    def set_owner(self, name, note=""):
        """登记属主（正在对话的那个人）。写入方用它做「我」的确定性消解。"""
        self.con.execute(
            "INSERT INTO meta(key,value) VALUES('owner',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (name,))
        if note:
            self.con.execute(
                "INSERT INTO meta(key,value) VALUES('owner_note',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (note,))

    def get_owner(self):
        row = self.con.execute("SELECT value FROM meta WHERE key='owner'").fetchone()
        return row["value"] if row else None

    # ------------------------------------------------ 通用元信息
    def set_meta(self, key, value):
        self.con.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))

    def get_meta(self, key):
        row = self.con.execute("SELECT value FROM meta WHERE key=?",
                               (key,)).fetchone()
        return row["value"] if row else None

    # ------------------------------------------------ 标定（只插不改）
    def calib_set(self, scope, key, k, v, note=None, fitted_t=0.0):
        self.con.execute(
            "INSERT INTO calib(scope,key,k,v,note,fitted_t) VALUES(?,?,?,?,?,?)",
            (scope, key, k, float(v), note, float(fitted_t)))

    def calib_latest(self, scope, key, k):
        row = self.con.execute(
            "SELECT v FROM calib WHERE scope=? AND key=? AND k=? "
            "ORDER BY fitted_t DESC, id DESC LIMIT 1", (scope, key, k)).fetchone()
        return row[0] if row else None

    # ------------------------------------------------ 图持久层
    def add_node(self, kind, label, t=None, salience=0.5, ref_table=None, ref_id=None):
        cur = self.con.execute(
            "INSERT INTO node(kind,label,t,salience,ref_table,ref_id) VALUES(?,?,?,?,?,?)",
            (kind, label, t, salience, ref_table, ref_id))
        return cur.lastrowid

    def find_node(self, kind, label):
        row = self.con.execute("SELECT id FROM node WHERE kind=? AND label=?",
                               (kind, label)).fetchone()
        return row[0] if row else None

    def node_ref(self, node_id):
        return self.con.execute(
            "SELECT kind,label,ref_table,ref_id FROM node WHERE id=?", (node_id,)).fetchone()

    def add_edge(self, a, b, kind, w, src=None):
        self.con.execute(
            "INSERT INTO edge(a,b,kind,w,src) VALUES(?,?,?,?,?) "
            "ON CONFLICT(a,b,kind) DO UPDATE SET w=MAX(w,excluded.w)",
            (a, b, kind, float(w), src))

    def edges_all(self):
        return self.con.execute("SELECT a,b,kind,w FROM edge").fetchall()

    def node_count(self):
        return self.con.execute("SELECT COUNT(*) FROM node").fetchone()[0]

    # ------------------------------------------------ 问答日志（冲突检测器 c5 用）
    def log_qa(self, q, t, outcome):
        self.con.execute("INSERT INTO qa_log(q,t,outcome) VALUES(?,?,?)", (q, t, outcome))

    def last_wrong_qa(self, q, window=None):
        row = self.con.execute(
            "SELECT outcome FROM qa_log WHERE q=? ORDER BY t DESC LIMIT 1", (q,)).fetchone()
        return row[0] if row else None

    # ------------------------------------------------ 杂
    def commit(self):
        self.con.commit()
        self.con.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def retract_fact(self, fact_id, t=None):
        """撤回一条事实（记忆修正的撤回原语）：status=retracted + valid_to=now。

        撤回本身是双时态事件：valid 盖到现在之后，asof 点查自然排除；
        回放撤回之前的历史仍能还原「当时确实知道过」。obs 原话永远在。
        返回是否真的撤了（已撤过/不存在 -> False）。
        """
        import time as _time
        now = float(t) if t is not None else _time.time()
        cur = self.con.execute(
            "UPDATE fact SET status='retracted', valid_to=COALESCE(valid_to,?) "
            "WHERE fact_id=? AND status != 'retracted'", (now, fact_id))
        return cur.rowcount > 0

    def retracted_count(self):
        return self.con.execute(
            "SELECT COUNT(*) FROM fact WHERE status='retracted'").fetchone()[0]

    def close(self):
        if getattr(self, "_closed", False):
            return                      # 幂等：主线程与关闭线程都会来关
        self._closed = True
        # 关库前把 pending 送进索引：不然这次会话新写的话
        # 下次启动要靠 sync 整目录重建才能查到（几十秒）
        try:
            self.flush_pending()
        except Exception:               # noqa: BLE001
            pass
        self.con.commit()
        try:
            self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass
        self.con.close()


def pack_vec(v):
    return struct.pack("%df" % len(v), *v)
