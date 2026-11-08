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

from . import config

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
  payload  TEXT
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
        self._fts = True
        try:
            self.con.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS obs_fts USING fts5("
                "payload, content='obs', content_rowid='obs_id')")
        except sqlite3.OperationalError:
            self._fts = False        # 没有 FTS5 就靠全表扫，不挡功能

    # ------------------------------------------------ 观测
    def add_obs(self, t, kind, channel, payload, entity=None, source=None,
                rel=2, seg_bnd=0.0, valence=None):
        cur = self.con.execute(
            "INSERT INTO obs(t,entity,kind,channel,source,rel,seg_bnd,valence,payload) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (t, entity, kind, channel, source, int(rel), float(seg_bnd),
             None if valence is None else float(valence), payload))
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
                pass                                    # 本人宣布改回来 -> 落到插入
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
        return cur.lastrowid, "new"

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
    def put_vec(self, kind, row_id, vec):
        v = np.asarray(vec, dtype=np.float32)
        self.con.execute(
            "INSERT INTO vec(kind,row_id,dim,v) VALUES(?,?,?,?) "
            "ON CONFLICT(kind,row_id) DO UPDATE SET dim=excluded.dim,v=excluded.v",
            (kind, row_id, int(v.size), v.tobytes()))

    def load_matrix(self, kind):
        """整块读出成 (ids, matrix)。矩阵常驻内存，检索时暴力扫 [STORE §4]。"""
        rows = self.con.execute("SELECT row_id,dim,v FROM vec WHERE kind=?", (kind,)).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        dim = rows[0][1]
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        mat = np.vstack([np.frombuffer(r[2], dtype=np.float32, count=dim) for r in rows])
        return ids, mat

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
        self.con.commit()
        try:
            self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass
        self.con.close()


def pack_vec(v):
    return struct.pack("%df" % len(v), *v)
