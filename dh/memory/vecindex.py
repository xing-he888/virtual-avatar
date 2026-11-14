# -*- coding: utf-8 -*-
"""向量索引层：Chroma 存向量（可重建），SQLite 存真值（权威）。

分工必须一句话说清，否则两边会互相打架：

    data/xiaoxing.db   ← **唯一真值**。obs/fact/event/vec 全在这里。
                         删了 Chroma 目录，重开就按 vec 表重建，一行资料不丢。
    data/chroma/       ← **派生索引**。只为了"几毫秒内找出最近邻"，
                         任何时刻都可以整目录删掉重建。

为什么用 Chroma 而不是继续 numpy 暴力扫：
    20 万条以内 numpy 扫一遍 5-17 ms，对话里够用；但 `recall()` 每次要扫
    fact + mention 两张表，还要给图扩展当种子，实测真实库 1679 条 mention
    已经是 6 ms，全量 20 万条会到 30 ms 以上，进入 LLM 首字延迟的预算里。
    HNSW 查询是亚毫秒级，且**增量插入不用重建整个矩阵**（numpy 路径每次
    新写一条 fact 都要 vstack 整个矩阵——20 万条时这一下就是几十毫秒）。

三条一致性纪律：
    1. **模型必须对齐**。collection 名里带模型名（`mem_fact__bge-small-zh-v1.5`），
       换模型等于换 collection，老的不删也不读——不同模型的余弦不可比。
    2. **启动时对齐数量**。`sync()` 比对 Chroma 的 count 和 SQLite 里该 kind
       该 model 的行数；不等就整 collection 重建。这条能兜住"进程被强杀，
       vec 写进去了但 Chroma 没落盘"的半截状态。
    3. **Chroma 报错不能拖垮检索**。任何异常都降级回 numpy 暴力扫
       （`retrieve.semantic_topk` 里有兜底），语义路变慢但不会瞎。
"""
import os
import threading

import numpy as np

from .. import config
from . import embed as embed_mod

# collection 名前缀：kind 和模型各占一段，换模型自然隔离
_PREFIX = "mem"


def default_dir(db_path):
    """索引目录跟着库走：同目录下的 chroma/ 子目录。"""
    if not db_path or db_path == ":memory:":
        return None
    base = os.path.dirname(os.path.abspath(db_path))
    return os.path.join(base, "chroma")


class VectorIndex:
    """一层薄封装。只做 upsert / query / count / sync，不做业务判断。"""

    def __init__(self, path, model=None):
        self.path = path
        self.model = model or embed_mod.HASH_NAME
        self._client = None
        self._lock = threading.Lock()
        self._err = None

    # ------------------------------------------------ 生命周期
    def _client_or_none(self):
        if self._client is not None:
            return self._client if self._client is not False else None
        with self._lock:
            if self._client is not None:
                return self._client if self._client is not False else None
            try:
                import chromadb
                from chromadb.config import Settings
                os.makedirs(self.path, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.path,
                    settings=Settings(anonymized_telemetry=False))
            except Exception as e:               # 没装 / 版本不对 / 目录不可写
                self._err = "%s: %s" % (type(e).__name__, e)
                self._client = False
        return self._client if self._client is not False else None

    def error(self):
        return self._err

    def available(self):
        return self._client_or_none() is not None

    def _col(self, kind):
        c = self._client_or_none()
        if c is None:
            return None
        name = "%s_%s__%s" % (_PREFIX, kind, self.model)
        # collection 名只允许 [a-zA-Z0-9._-]，模型名里的斜杠要换掉
        name = name.replace("/", "-")
        try:
            return c.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"})
        except Exception as e:
            self._err = "%s: %s" % (type(e).__name__, e)
            return None

    # ------------------------------------------------ 写
    def upsert(self, kind, rows):
        """rows = [(row_id, vector)]。失败静默（下次 sync 会补）。"""
        col = self._col(kind)
        if col is None or not rows:
            return 0
        ids = [str(int(r[0])) for r in rows]
        embs = [np.asarray(r[1], dtype=np.float32).tolist() for r in rows]
        try:
            col.upsert(ids=ids, embeddings=embs,
                       metadatas=[{"kind": kind} for _ in ids])
            return len(ids)
        except Exception as e:
            self._err = "%s: %s" % (type(e).__name__, e)
            return 0

    # ------------------------------------------------ 读
    def query(self, kind, vector, k=20):
        """返回 [(row_id, cosine_sim)]，已按相似度降序。失败返回 []。"""
        col = self._col(kind)
        if col is None:
            return []
        try:
            r = col.query(query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
                          n_results=max(1, int(k)),
                          include=["distances"])
        except Exception as e:
            self._err = "%s: %s" % (type(e).__name__, e)
            return []
        ids = (r.get("ids") or [[]])[0]
        dist = (r.get("distances") or [[]])[0]
        # cosine space 里 distance = 1 - cos，换回余弦
        return [(int(i), float(1.0 - d)) for i, d in zip(ids, dist)]

    def count(self, kind):
        col = self._col(kind)
        if col is None:
            return -1
        try:
            return int(col.count())
        except Exception:
            return -1

    # ------------------------------------------------ 对齐
    def sync(self, store, kinds=None, batch=512):
        """把 SQLite 的 vec 表同步进 Chroma。数量一致就跳过。

        kinds 默认取 config.VEC_INDEX_KINDS（不是写死元组）——
        检索侧扫哪些 kind 由同一份配置决定，两边只有一个真源。

        返回 dict(kind -> dict(sqlite=N, chroma=M, rebuilt=bool))。
        """
        kinds = kinds or config.VEC_INDEX_KINDS
        out = {}
        for kind in kinds:
            ids, mat = store.load_matrix(kind, model=self.model)
            n_sql = int(len(ids))
            n_ix = self.count(kind)
            info = dict(sqlite=n_sql, chroma=n_ix, rebuilt=False)
            if n_ix == n_sql:
                out[kind] = info
                continue
            col = self._col(kind)
            if col is None:
                out[kind] = info
                continue
            try:
                # 整 collection 重建：比逐条 diff 简单，且这条路径只在
                # 「进程被强杀 / 换模型 / 首次接入」时走一次
                c = self._client_or_none()
                name = col.name
                c.delete_collection(name)
                col = c.create_collection(name=name,
                                          metadata={"hnsw:space": "cosine"})
                for i in range(0, n_sql, batch):
                    part_ids = ids[i:i + batch]
                    part = mat[i:i + batch]
                    col.add(ids=[str(int(x)) for x in part_ids],
                            embeddings=part.tolist(),
                            metadatas=[{"kind": kind} for _ in part_ids])
                info["rebuilt"] = True
                info["chroma"] = n_sql
            except Exception as e:
                self._err = "%s: %s" % (type(e).__name__, e)
            out[kind] = info
        return out
