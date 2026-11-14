# -*- coding: utf-8 -*-
"""嵌入层：真实语义向量（本地 bge-small-zh），带 hash 兜底。

为什么换掉 hash_embed：
    hash_embed 是 char 2-gram 的稳定哈希——它只认**字面重叠**。
    「老王住院了」vs「老王在住院」高（共享字），
    「老王住院了」vs「老王进医院了」≈0（"住院"/"医院"哈希到不同维）。
    实测 bge-small-zh-v1.5：0.8809 / 0.9695，这才是语义。

三条纪律：
    1. **不引 sentence_transformers**。用 transformers 直接跑 CLS pooling，
       模型 95 MB，CPU 单句 2.2 ms，跟仓库一起走，不联网。
    2. **模型不在就降级**。别人 clone 后没下模型也能跑（退回 hash_embed），
       只是语义路退化成字面路——不报错、不挡功能。
    3. **确定性的口径变了**。hash_embed 是纯函数；神经网络嵌入有浮点非确定性。
       所以向量写进 vec 表就当成缓存，运行时只读库、不再重算——
       复现只依赖库文件，不依赖 torch 版本。
"""
import math
import os
import threading

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models", "bge-small-zh-v1.5")

# 模型标识：写进 vec.model，换模型就整表重算（不同模型的余弦不可比）
MODEL_NAME = "bge-small-zh-v1.5"
HASH_NAME = "hash256"

_lock = threading.Lock()
_loaded = None          # (tokenizer, model) 或 False（加载失败过，不再重试）
_load_error = None


def model_available(path=None):
    """模型目录在不在（三个必需文件齐全）。"""
    p = path or MODEL_DIR
    need = ("config.json", "model.safetensors", "tokenizer.json")
    return all(os.path.exists(os.path.join(p, f)) for f in need)


def _load(path=None):
    """懒加载 + 单次失败不重试（避免每条记忆都去碰一次坏路径）。"""
    global _loaded, _load_error
    if _loaded is not None:
        return _loaded
    with _lock:
        if _loaded is not None:
            return _loaded
        p = path or MODEL_DIR
        if not model_available(p):
            _load_error = "model files missing at %s" % p
            _loaded = False
            return _loaded
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
            tok = AutoTokenizer.from_pretrained(p)
            mdl = AutoModel.from_pretrained(p)
            mdl.eval()
            _loaded = (tok, mdl)
        except Exception as e:              # 权重损坏 / 版本不兼容 / 没 torch
            _load_error = "%s: %s" % (type(e).__name__, e)
            _loaded = False
    return _loaded


def load_error():
    return _load_error


def bge_embed(text, path=None):
    """CLS pooling + L2 归一。返回 float32 一维数组；失败返回 None。"""
    m = _load(path)
    if not m:
        return None
    tok, mdl = m
    import torch
    with torch.no_grad():
        b = tok([text], padding=True, truncation=True, max_length=128,
                return_tensors="pt")
        out = mdl(**b).last_hidden_state[:, 0]
    v = out[0].numpy().astype(np.float32)
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def bge_embed_batch(texts, path=None, batch_size=32):
    """批量嵌入（摄取时用）。返回 (N, dim) float32，已归一。"""
    m = _load(path)
    if not m:
        return None
    tok, mdl = m
    import torch
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            part = list(texts[i:i + batch_size])
            b = tok(part, padding=True, truncation=True, max_length=128,
                    return_tensors="pt")
            out = mdl(**b).last_hidden_state[:, 0].numpy().astype(np.float32)
            chunks.append(out)
    if not chunks:
        return np.zeros((0, 512), dtype=np.float32)
    mat = np.vstack(chunks)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


_HASH_DIM = 256


def hash_embed(text, dim=_HASH_DIM):
    """无网络兜底：char 2-gram 稳定哈希（crc32，不用内置 hash）。"""
    import zlib
    v = np.zeros(dim, dtype=np.float32)
    s = "".join(ch for ch in text if not ch.isspace())
    for i in range(len(s) - 1):
        v[zlib.crc32(("%s|%s" % (s[i], s[i + 1])).encode("utf-8")) % dim] += 1.0
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def make_embed_fn(prefer_model=True, path=None):
    """工厂：模型在就用 bge，不在就退回 hash。

    返回 (embed_fn, name)。name 要写进 vec.model —— 不同模型的余弦不可比，
    换了模型必须整表重算，否则算出来的相似度是噪声。
    """
    if prefer_model and model_available(path) and _load(path):
        # 同一句话反复说（世界里 1800 轮只有 737 句唯一）——不缓存的话
        # 每条 obs 都跑一遍前向，摄取慢十几倍。纯函数，缓存安全
        cache = {}

        def _fn(t):
            hit = cache.get(t)
            if hit is not None:
                return hit
            v = bge_embed(t, path)
            if v is None:
                return np.zeros(512, dtype=np.float32)
            if len(cache) < 20000:
                cache[t] = v
            return v
        _fn.model_name = MODEL_NAME
        return _fn, MODEL_NAME
    hash_embed.model_name = HASH_NAME
    return hash_embed, HASH_NAME


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size != b.size or a.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
