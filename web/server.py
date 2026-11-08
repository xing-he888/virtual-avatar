# -*- coding: utf-8 -*-
"""小星聊天页面的后端：静态文件 + 三个 JSON 接口。

    GET  /api/state     小星此刻的情绪（PAD + 离散标签 + 表情 + 韵律）与自我叙事
    POST /api/chat      说话 -> 听见(hear) -> 记忆点查 or LLM -> 回复 + 情绪快照
    GET/POST /api/settings  API Key 配置（只落本机 web/api_settings.local.json，不进 git）
    POST /api/settings/test    用一条 1-token 请求验证连通

情绪呈现的三个调研依据（dh/IMPLEMENTATION.md 同款出处风格）：
    Valdez & Mehrabian 1994  亮度/饱和度驱动唤醒感知，色相主导愉悦响应
                             -> 效价映射色相（暖橙/冷蓝），唤醒映射辉光
    Emerald 2024 "golden zone" 情绪表达过度反而失真 -> 表达克制：状态条+微色变
    Pezenka 2024 / USF 2022  情绪表达提升拟人化与温暖 -> 回复节奏是文字聊天的韵律通道

运行：
    C:/Users/xinghe/miniconda3/envs/virtual_avatar/python.exe web/server.py
    浏览器打开 http://127.0.0.1:8787
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dh import DigitalHuman, MemoryStore  # noqa: E402
from dh import config  # noqa: E402
from dh.llm_bridge import build_system_prompt, offline_reply  # noqa: E402
from dh.tick import looks_like_question  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "api_settings.local.json")
STATIC = os.path.join(HERE, "static")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"

# PAD 离散标签锚点（Mehrabian & Russell 1974 情绪集 / Russell 1980 环状模型，
# 坐标取 [-1,1] 近似值）。设计纪律：内部永远是向量，离散标签只在输出时生成。
PAD_ANCHORS = [
    ("喜悦", (0.8, 0.3, 0.3)), ("兴奋", (0.6, 0.8, 0.2)),
    ("满足", (0.7, -0.3, 0.4)), ("平静", (0.3, -0.5, 0.1)),
    ("自豪", (0.4, 0.4, 0.8)), ("无聊", (-0.3, -0.6, -0.2)),
    ("低落", (-0.7, -0.4, -0.3)), ("沮丧", (-0.8, -0.1, -0.4)),
    ("焦虑", (-0.5, 0.6, -0.7)), ("恼火", (-0.6, 0.7, -0.3)),
    ("紧张", (-0.3, 0.5, -0.6)),
]


def pad_label(p, a, d):
    """离散标签 = PAD 空间最近邻。靠近原点一律叫平静（黄金区间：别过度表达）。"""
    import math
    if math.sqrt(p * p + a * a + d * d) < 0.15:
        return "平静"
    best, bd = "平静", 1e9
    for name, (ap, aa, ad) in PAD_ANCHORS:
        dist = (p - ap) ** 2 + (a - aa) ** 2 + (d - ad) ** 2
        if dist < bd:
            best, bd = name, dist
    return best


def pad_emoji(p, a):
    """表情读 A+P（DESIGN §4.3 表情出口）。"""
    if p > 0.3:
        return "😄" if a > 0.3 else "😊"
    if p < -0.3:
        return "😠" if a > 0.5 else "😔"
    if a > 0.5:
        return "😯"
    return "🙂"


def mood_snapshot(dh):
    st = dh.affect.state
    p, a, d = st.p, st.a, st.d
    return dict(p=round(p, 3), a=round(a, 3), d=round(d, 3),
                label=pad_label(p, a, d), emoji=pad_emoji(p, a),
                rate=dh.affect.outlets()["speech"]["rate"],
                lang=dh.affect.outlets()["language"])


class LLMClient:
    """DashScope OpenAI 兼容接口的薄封装。Key 优先级：本机配置 > .env。"""

    def __init__(self, settings_path):
        self.settings_path = settings_path

    def _cfg(self):
        cfg = dict(api_key=None, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL)
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, encoding="utf-8") as f:
                    saved = json.load(f)
                cfg.update({k: saved[k] for k in ("api_key", "base_url", "model")
                            if saved.get(k)})
            except (OSError, ValueError):
                pass
        if not cfg["api_key"]:
            cfg["api_key"] = os.environ.get("ALIYUN_APIKEY")
        return cfg

    def configured(self):
        return bool(self._cfg()["api_key"])

    def masked(self):
        key = self._cfg()["api_key"]
        if not key:
            return dict(has_key=False, tail="", base_url=self._cfg()["base_url"],
                        model=self._cfg()["model"])
        return dict(has_key=True, tail=key[-4:], base_url=self._cfg()["base_url"],
                    model=self._cfg()["model"])

    def chat(self, system_prompt, user_text, timeout=60, temperature=None,
             max_tokens=16000):
        """max_tokens 给足——deepseek-v4.1-flash 是推理模型，思维链吃
        token，给 2000 会让它把预算花光在 reasoning 上、正文为空
        （finish=length，content 空）。另：content 空时回退 reasoning_content，
        抽取的 JSON 有时写在思维链里。"""
        cfg = self._cfg()
        if not cfg["api_key"]:
            raise RuntimeError("未配置 API Key")
        from openai import OpenAI
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"],
                        timeout=timeout)
        kw = dict(model=cfg["model"],
                  messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text}],
                  max_tokens=max_tokens)
        if temperature is not None:
            kw["temperature"] = temperature
        resp = client.chat.completions.create(**kw)
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        if not content:
            content = (getattr(msg, "reasoning_content", None) or "").strip()
        return content

    def test(self):
        try:
            out = self.chat("你是一个测试探针，只回答：ok", "ping", timeout=15)
            return dict(ok=True, sample=(out or "")[:40])
        except Exception as exc:                      # noqa: BLE001
            return dict(ok=False, error=str(exc)[:300])


class ChatServer:
    """把 DigitalHuman 单例挂到 HTTP 上。store/settings 路径可注入（测试用）。

    idle_seconds > 0 时起一个后台线程：空闲够久就跑第 7 步
    （巩固 + 心智游移），游移产出经 /api/state 推给前端主动开口。
    """

    def __init__(self, store_path="data/xiaoxing.db",
                 settings_path=SETTINGS_PATH, seed=0,
                 idle_seconds=None, extractor=None):
        os.makedirs(os.path.dirname(os.path.abspath(store_path)) or ".", exist_ok=True)
        self.dh = DigitalHuman(store=MemoryStore(store_path), seed=seed)
        self.llm = LLMClient(settings_path)
        self.settings_path = settings_path
        self.history = []          # 本页会话的消息（内存里，刷新不丢，重启丢）
        self.idle_seconds = (config.IDLE_WANDER_S if idle_seconds is None
                             else idle_seconds)
        self.last_activity = time.time()
        self.last_wander = None    # dict(t, thoughts, touched)
        self.activations = []      # 识海：真实发生的点亮事件（对话扩散/游移路径）
        self.last_user_text = ""
        self.extractor = extractor            # 可注入 Mock（测试）
        self._extract_busy = threading.Semaphore(1)
        self.last_late_extract = None
        self.last_cited = []                  # 上一轮回复引用的事实行（动态修正锚）
        self._stop = threading.Event()
        if self.idle_seconds and self.idle_seconds > 0:
            self._th = threading.Thread(target=self._wander_loop, daemon=True)
            self._th.start()

    def stop(self):
        """停掉游移线程（测试清理用）。join 等它退出，别跟关库赛跑。"""
        self._stop.set()
        th = getattr(self, "_th", None)
        if th is not None and th.is_alive():
            th.join(timeout=2.0)

    def shutdown(self):
        """优雅关闭：停线程 + 提交合并 WAL + 关库。/api/shutdown 调用。"""
        self.stop()
        try:
            self.dh.store.close()      # close 内含 commit + TRUNCATE checkpoint
        except Exception:              # noqa: BLE001
            pass

    def _wander_loop(self):
        while not self._stop.is_set():
            time.sleep(1.0)
            if time.time() - self.last_activity < self.idle_seconds:
                continue
            try:
                with self.dh.store.lock:
                    res = self.dh.idle(t=time.time())
            except Exception:                          # noqa: BLE001
                self.last_activity = time.time()
                continue
            self.last_activity = time.time()           # 游移完重新计时
            w = res.get("wander") or {}
            if w.get("thoughts"):
                self.last_wander = dict(t=time.time(),
                                        thoughts=w["thoughts"][:3],
                                        touched=w.get("touched", 0))
                if w.get("seen"):
                    acc = {k: min(1.0, v / 3.0) for k, v in w["seen"].items()}
                    self.push_activation("游移", acc)

    # ------------------------------------------------ 联想记忆
    def on_associate(self, text):
        """「提到X，还会想起什么」：激活扩散 + 联想循环（ρ/K/抽样）。

        种子 = 话题里抽到的实体名下的记忆（GRAPH §4 第 3 条：先 SQL 挑
        记忆再扩散，不从实体节点出发）。
        """
        with self.dh.store.lock:
            dh = self.dh
            parsed = dh.ex.parse(text) or dh.ex.parse_question(text)
            ents = [e[2] for e in parsed["entities"]] if parsed else []
            if not ents:
                ents = [k for k in dh.ex.alias if k in text]
            fids = []
            if ents:
                q = ("SELECT DISTINCT fact_id FROM fact WHERE entity IN (%s)"
                     % ",".join("?" * len(ents)))
                fids = [r[0] for r in dh.store.con.execute(q, ents).fetchall()]
            if not fids:
                return dict(text="「%s」这个话题我还没记过什么，联想不起来。"
                                 "先跟我说说相关的事吧。" % text[:16],
                            kind="associate", touched=0, repeat_rate=0.0,
                            sources=[], used_memory=False, offline=True)
            seeds = {f: 1.0 for f in fids[:2]}
            res = dh.graph.associate(seeds, np.random.default_rng(), rounds=6,
                                     per_round=20)
            lines = []
            for i, picks in enumerate(res["picks"], 1):
                texts = []
                for fid in picks[:2]:
                    row = dh.store.fact_row(fid)
                    if row is None:
                        continue
                    obs = dh.store.obs_row(row["src_obs"])
                    if obs is not None and obs["payload"]:
                        texts.append(obs["payload"][:22])
                if texts:
                    lines.append("第%d轮想起：%s" % (i, " / ".join(texts)))
            reply = ("由「%s」顺着想下去（%d 条记忆被点亮）：\n%s"
                     % (text[:16], res["touched"], "\n".join(lines[:6])))
            return dict(text=reply, kind="associate", touched=res["touched"],
                        repeat_rate=res["repeat_rate"], sources=[],
                        used_memory=True, offline=True)

    # ------------------------------------------------ 对话主流程
    # ------------------------------------------------ 动态记忆修正（聊天流内）
    # 修正信号词：一句话以这些开头，且上一轮回复引用过某条事实 -> 视为修正
    # （「我/你」可作前缀：「我记错了，不对，…」）
    CORRECT_LEAD_RE = re.compile(
        r"^(?:[我你咱])?(不对|错了|说错了?|记错了?|不是这[样个]?|不是的?|"
        r"应该[是说]?|其实是?|好像是?|变成|换成|改成|现在是?|就是|是)"
        r"[，,。！!：:、\s]*")
    CORRECT_TAIL_RE = re.compile(r"[呀呢啦哦嘛了。！!？?]+$")

    def _dynamic_correction(self, text, hear_out=None):
        """动态记忆修正。两条路：

        关系路（无需锚）：这句带修正信号且 hear 刚写出/撤换了关系——
        「不对，小王是小李的弟弟」-> hear 已写 关系.弟弟 并把冲突的
        哥哥撤回，这里只负责确认回复。
        值路（需锚）：上一轮回复引用过一条事实 + 修正信号 -> 新值必须
        是这句话里逐字出现的连续片段（G3 的口语版：长数字/引号/短尾），
        LLM 只能指认不能发明。
        返回 dict(reply, fact_id, changed) 或 None（不是修正，走常规流）。
        """
        if not self.CORRECT_LEAD_RE.match(text):
            return None
        # 关系路：本轮 hear 已写关系事实（冲突旧关系已自动撤回）
        rel_facts = (hear_out.get("relation_facts") or []) if hear_out else []
        if rel_facts:
            lines = []
            for f in rel_facts:
                ent, pred = f["key"].split("|", 1)
                lines.append("%s的%s是%s" % (ent, pred.split(".", 1)[1],
                                             f["value"]))
            old = (hear_out.get("relations_retracted") or [])
            note = "".join("（原来记的「%s」已撤回）" % w for w, _ in old)
            return dict(reply="改好了：" + "；".join(lines) + note + "。",
                        fact_id=rel_facts[0]["fact_id"], changed=True)
        # 值路：锚定上一轮引用的那条
        if len(self.last_cited) != 1:
            return None                     # 没锚或多锚（多值）先不猜
        fid = self.last_cited[0]
        fact = self.dh.store.fact_row(fid)
        if fact is None or fact["status"] != "active":
            return None
        body = text
        for _ in range(3):                  # 「不对，其实是…」剥到内容为止
            m = self.CORRECT_LEAD_RE.match(body)
            if not m:
                break
            body = body[m.end():]
        body = self.CORRECT_TAIL_RE.sub("", body).strip()
        if not body:
            return None
        m = re.search(r"\d{5,}", body)      # 长数字（电话/QQ/学号…）
        if m:
            val = m.group(0)
        else:
            m = re.search(r"[「「『]([^」」』]+)[」」』]", body)
            val = m.group(1) if m else (body if 1 <= len(body) <= 20 else None)
        if not val:
            return None
        if val == fact["value"]:
            return dict(reply="这我本来就这么记的：%s的%s是%s。"
                              % (fact["entity"], fact["predicate"], val),
                        fact_id=fid, changed=False)
        self.correct(fid, value=val)
        return dict(reply="改好了：%s的%s是%s。原来是%s，已经记成新的了。"
                          % (fact["entity"], fact["predicate"], val,
                             fact["value"]),
                    fact_id=fid, changed=True)

    def on_chat(self, text):
        dh = self.dh
        with dh.store.lock:
            self.last_activity = time.time()
            self.last_user_text = text
            return self._on_chat_locked(text)

    def _on_chat_locked(self, text):
        dh = self.dh
        t = time.time()
        # 1 它先听见：感知/注意/评估/写回（情绪与记忆在此更新）
        hear_out = dh.hear(text, speaker="用户", t=t)
        # 1.5 动态记忆修正：关系修正（显式双名，无需锚）或上一轮引用过
        #     的事实被口头否定 -> 当场改记忆并确认，不进常规问答流
        corr = self._dynamic_correction(text, hear_out)
        if corr is not None:
            reply = corr["reply"]
            hear_out_reply = dh.hear(reply, speaker="小星", t=time.time() + 0.01)
            dh.store.commit()               # 修正即落盘
            rec = dict(role="user", text=text, t=t)
            rec2 = dict(role="xiaoxing", text=reply, mood=mood_snapshot(dh),
                        flags=[], sources=[], used_memory=True,
                        offline=not self.llm.configured(), stale=False,
                        cited=[], corrected=dict(fact_id=corr["fact_id"],
                                                 changed=corr["changed"]),
                        t=time.time())
            self.history.extend([rec, rec2])
            self.last_cited = []
            return rec2
        # 2 快路径：事实类**问句**点查记忆，不花 token（来源措辞现成）。
        # 带值的陈述句（「老王的电话是X」）不是问题，别抢答成回声。
        parsed = dh.ex.parse_question(text)
        fast = None
        if parsed is not None and looks_like_question(text):
            ans, phr, diag = dh.ask(text)
            if not ans.abstain:
                # 「来源：用户」没有信息量（L1 措辞本身就是「你告诉过我」）
                srcs = [s for s in (ans.sources or []) if s not in ("用户", "我")]
                fast = dict(reply=phr, flags=diag["conflict_flags"],
                            sources=srcs, stale=bool(ans.stale),
                            degraded=bool(ans.degraded), used_memory=True,
                            cited=list(ans.fact_ids))
        # 3 慢路径：LLM 组织语言（记忆上下文 + 情绪语气注入）
        if fast is not None:
            reply, flags, srcs = fast["reply"], fast["flags"], fast["sources"]
            used_memory = True
            stale = fast["stale"]
            self.last_cited = fast["cited"]   # 记下引用了哪条 -> 下一句可动态修正
        elif self.llm.configured():
            prompt = build_system_prompt(dh, text)
            reply = self.llm.chat(prompt, text)
            reply = (reply or "").strip()
            flags, srcs, used_memory, stale = [], [], True, False
            self.last_cited = []              # LLM 路径没有确切引用锚
        else:
            reply, meta = offline_reply(dh, text)
            flags, srcs = [], []
            used_memory, stale = meta["used_memory"], False
            self.last_cited = []
        # 4 自己说的话也记（self 的陈述 rel=1）
        hear_out_reply = dh.hear(reply, speaker="小星", t=time.time() + 0.01)
        # 5 识海：本次对话点亮了哪些记忆 -> 真实扩散一次，激活图交给前端播放
        try:
            touched = [fid for out in (hear_out, hear_out_reply)
                       for fid in out.get("written", [])]
            if touched:
                acc = dh.graph.diffuse({f: 1.0 for f in touched})
                self.push_activation("对话", acc)
        except Exception:                              # noqa: BLE001
            pass
        snap = mood_snapshot(dh)
        # offline 的语义 = 本条回复没走 LLM（快路径点查 / 离线兜底都算）
        offline = not self.llm.configured()
        rec = dict(role="user", text=text, t=t)
        rec2 = dict(role="xiaoxing", text=reply, mood=snap, flags=flags,
                    sources=srcs, used_memory=used_memory, offline=offline,
                    stale=stale, cited=list(self.last_cited), t=time.time())
        self.history.extend([rec, rec2])
        self.dh.store.commit()      # 消息即落盘（commit 后 PASSIVE checkpoint）
        # LLM 补记：正则没抽出事实、但句子可能有信息 -> 后台抽取（不阻塞回复）
        if not hear_out.get("written") and (self.extractor is not None
                                            or self.llm.configured()):
            threading.Thread(target=self._late_extract,
                             args=(text, t, "用户", hear_out), daemon=True).start()
        return rec2

    def _late_extract(self, text, t, speaker, hear_out):
        """LLM 补记：五道闸门 -> 写回 -> 识海点亮「回味」。失败静默（obs 永远在）。"""
        if not self._extract_busy.acquire(blocking=False):
            return                               # 上一条还没抽完，丢给下一条
        try:
            ex = self.extractor
            if ex is None and self.llm.configured():
                from dh.extract_llm import LLMExtractor, default_registry
                ex = LLMExtractor(
                    lambda sys_p, usr: self.llm.chat(sys_p, usr, temperature=0.1),
                    registry=default_registry(), extractor=self.dh.writer.ex)
                self.extractor = ex
            if ex is None:
                return
            topics = [it["text"] for it in self.dh.wm.items[-2:]]
            cands, _ = ex.extract(text, topics)
            if not cands:
                return
            with self.dh.store.lock:
                obs_ids = hear_out.get("obs_ids") or []
                obs_id = obs_ids[0] if obs_ids else None
                if obs_id is None:               # 问句路径没有 writer hear——补一条 obs
                    obs_id = self.dh.store.add_obs(t, "told", "text", text,
                                                   source=speaker, rel=1, seg_bnd=0.0)
                def _learn(name):
                    if self.dh.writer.ex.learn_alias(name):
                        self.dh.store.alias_learn(name, name)
                out = self.dh.writer.hear_llm_candidates(
                    text, t, speaker, obs_id, cands, learn_entity_cb=_learn)
                acc = self.dh.graph.diffuse({f: 1.0 for f in out["fact_ids"]})                     if out["fact_ids"] else {}
            if out["fact_ids"]:
                self.dh.store.commit()       # 后台写入自己落盘
                self.push_activation("补记", acc)
                self.last_late_extract = dict(t=time.time(),
                                              fact_ids=out["fact_ids"],
                                              actions=out["actions"])
        except Exception:                        # noqa: BLE001
            pass                                 # obs 里原文永远在，丢了可重来
        finally:
            self._extract_busy.release()

    # ------------------------------------------------ 识海（记忆星图）
    def display_graph(self):
        """识海页的图：实体枢纽 + 记忆星 + 三类边（share/link/sim）。

        这是真值层的忠实投影：share 边权沿用 D/n 摊薄，sim 边按余弦
        取每星前 4 条（防头发团）。结果按 fact 数缓存，库不变不重算。
        """
        dh = self.dh
        with dh.store.lock:
            n_fact = dh.store.fact_count()
            n_mention = dh.store.con.execute(
                "SELECT COUNT(*) FROM mention").fetchone()[0]
            cache = getattr(self, "_graph_cache", None)
            # 缓存键看四个数：fact / mention（link 边）/ 带实体的 obs（co 边
            # 原料）/ 撤回数（修正不改变 fact 总数，必须单独看）
            n_ent = dh.store.con.execute(
                "SELECT COUNT(*) FROM obs WHERE entity IS NOT NULL").fetchone()[0]
            n_ret = dh.store.retracted_count()
            if cache is not None and cache[0] == (n_fact, n_mention, n_ent, n_ret):
                return cache[1]
            nodes, edges = [], []
            seen_edges = set()
            # 撤回（记忆修正）的事实不进星图——「忘了」就从海里消失
            grp = dh.store.con.execute(
                "SELECT entity, COUNT(*) AS n FROM fact "
                "WHERE status != 'retracted' GROUP BY entity").fetchall()
            deg = {r["entity"]: r["n"] for r in grp}
            # 纯关系实体（只作为关系的 value 出现、自己名下暂无事实）——
            # 老王的父亲这种「关系提到的亲人」也是星图上的一等枢纽
            for r in dh.store.con.execute(
                    "SELECT DISTINCT value FROM fact "
                    "WHERE predicate LIKE '关系.%'").fetchall():
                if r["value"] and r["value"] not in deg:
                    deg[r["value"]] = 0
            for r in grp:
                nodes.append(dict(id="e:" + r["entity"], kind="entity",
                                  label=r["entity"], degree=r["n"], valence=None))
            for name, n in deg.items():
                if n == 0:      # rel-only 枢纽补节点
                    nodes.append(dict(id="e:" + name, kind="entity",
                                      label=name, degree=0, valence=None))
            rows = dh.store.con.execute(
                "SELECT fact_id, entity, predicate, value, src_obs, written_t "
                "FROM fact WHERE status != 'retracted' ORDER BY fact_id").fetchall()
            for r in rows:
                obs = dh.store.obs_row(r["src_obs"])
                val = None
                tip = ""
                if obs is not None:
                    tip = obs["payload"] or ""
                    if obs["valence"] is not None:
                        val = float(obs["valence"])
                    elif tip:
                        from dh.extract import Extractor
                        val = Extractor.valence(tip)
                nodes.append(dict(id="f:%d" % r["fact_id"], kind="fact",
                                  entity=r["entity"],
                                  label="%s=%s" % (r["predicate"], r["value"]),
                                  tip=tip, valence=val, t=r["written_t"],
                                  degree=1))
                w = config.EDGE_W["share"] * min(1.0, config.HUB_DEG_CAP /
                                                 max(deg.get(r["entity"], 1), 1))
                edges.append(dict(a="e:" + r["entity"], b="f:%d" % r["fact_id"],
                                  kind="share", w=round(w, 4)))
                seen_edges.add(("e:" + r["entity"], "f:%d" % r["fact_id"]))
            # link：同一事件牌子下的说法互连（零共同词靠它）
            obs2fact = {r["src_obs"]: r["fact_id"] for r in rows}
            pairs = dh.store.con.execute(
                "SELECT m1.obs_id AS oa, m2.obs_id AS ob FROM mention m1 "
                "JOIN mention m2 ON m2.event_id=m1.event_id "
                "AND m2.mention_id<>m1.mention_id").fetchall()
            for r in pairs:
                a, b = obs2fact.get(r["oa"]), obs2fact.get(r["ob"])
                if a is None or b is None or a >= b:
                    continue
                key = ("f:%d" % a, "f:%d" % b)
                if key not in seen_edges:
                    edges.append(dict(a=key[0], b=key[1], kind="link",
                                      w=config.EDGE_W["link"]))
                    seen_edges.add(key)
            # rel：关系事实（关系.X=对方实体）→ 实体-实体边（识海里把星团连起来）
            known_ents = {"e:" + r["entity"] for r in grp} | {
                "e:" + n["label"] for n in nodes if n["kind"] == "entity"}
            for r in dh.store.con.execute(
                    "SELECT entity, predicate, value FROM fact "
                    "WHERE predicate LIKE '关系.%' AND status != 'retracted'").fetchall():
                hub_a, hub_b = "e:" + r["entity"], "e:" + r["value"]
                if hub_a in known_ents and hub_b in known_ents and hub_a != hub_b:
                    key = tuple(sorted([hub_a, hub_b]))
                    if key not in seen_edges:
                        edges.append(dict(a=key[0], b=key[1], kind="rel", w=0.9))
                        seen_edges.add(key)
            # co：同一条原话里出现的多个实体（零成本兜底——RAG 思路的规则版）
            for r in dh.store.con.execute(
                    "SELECT obs_id, entity FROM obs "
                    "WHERE entity IS NOT NULL AND entity LIKE '%,%'").fetchall():
                ents = [e for e in r["entity"].split(",") if e]
                for i in range(len(ents)):
                    for j in range(i + 1, len(ents)):
                        a, b = "e:" + ents[i], "e:" + ents[j]
                        if a in known_ents and b in known_ents:
                            key = tuple(sorted([a, b]))
                            if key not in seen_edges:
                                edges.append(dict(a=key[0], b=key[1], kind="co",
                                                  w=0.3))
                                seen_edges.add(key)
            # sim：向量近邻，每星只取最强 4 条
            ids, mat = dh.store.load_matrix("fact")
            if len(ids) >= 2 and mat.shape[1] > 0:
                nrm = np.linalg.norm(mat, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                m = mat / nrm
                sim = m @ m.T
                for i in range(len(ids)):
                    cand = [(sim[i, j], j) for j in range(len(ids))
                            if j != i and sim[i, j] >= config.SIM_EDGE_MIN_COS]
                    for v, j in sorted(cand, reverse=True)[:4]:
                        a, b = int(ids[i]), int(ids[j])
                        key = ("f:%d" % min(a, b), "f:%d" % max(a, b))
                        if key not in seen_edges:
                            edges.append(dict(a=key[0], b=key[1], kind="sim",
                                              w=round(config.EDGE_W["sim"] * float(v), 4)))
                            seen_edges.add(key)
            out = dict(version=n_fact + n_ret, nodes=nodes, edges=edges)
            self._graph_cache = ((n_fact, n_mention, n_ent, n_ret), out)
            return out

    def push_activation(self, name, acc):
        """记一次真实发生的点亮（对话扩散 / 游移路径），识海播放。
        键统一成 "f:<fact_id>"（前端按节点 id 查找）。"""
        acc = {"f:%s" % k: round(float(v), 4) for k, v in acc.items() if v > 0.005}
        if not acc:
            return
        self.activations.append(dict(t=time.time(), name=name, acc=acc))
        del self.activations[:-4]

    def state(self):
        with self.dh.store.lock:
            return self._state_locked()

    def _state_locked(self):
        dh = self.dh
        n_active = dh.store.con.execute(
            "SELECT COUNT(*) FROM fact WHERE status='active'").fetchone()[0]
        return dict(mood=mood_snapshot(dh),
                    narrs=[r[1] for r in dh.store.self_narrs()],
                    wander=self.last_wander,
                    activations=list(self.activations),
                    llm=self.llm.masked(),
                    n_facts=n_active,
                    n_obs=dh.store.obs_count())

    # ------------------------------------------------ 记忆修正
    def memories(self):
        """修正面板的数据：全部未撤回的事实，带原文出处与来源分级。"""
        with self.dh.store.lock:
            rows = self.dh.store.con.execute(
                "SELECT f.fact_id, f.entity, f.predicate, f.value, f.status, "
                "f.rel, f.conf, f.valid_from, o.source, o.payload "
                "FROM fact f LEFT JOIN obs o ON o.obs_id=f.src_obs "
                "ORDER BY f.entity, f.predicate, f.fact_id").fetchall()
            out = [dict(fact_id=r["fact_id"], entity=r["entity"],
                        predicate=r["predicate"], value=r["value"],
                        status=r["status"], rel=int(r["rel"] or 2),
                        conf=r["conf"], t=r["valid_from"],
                        source=r["source"] or "",
                        obs=(r["payload"] or "")[:80])
                   for r in rows]
            return dict(memories=out)

    def correct(self, fact_id, entity=None, predicate=None, value=None):
        """修正一条记忆——写「修正后的事实」，双时态存储自己料理旧账：

        只改值：write_fact 正常取代（旧行 superseded，历史保留）；
        改主体/谓词：旧的三元组整个记错 -> 撤回（retracted），新事实另写；
        多值谓词改值：修正=换掉这一条，旧行撤回。
        """
        with self.dh.store.lock:
            old = self.dh.store.con.execute(
                "SELECT * FROM fact WHERE fact_id=?", (fact_id,)).fetchone()
            if old is None:
                return dict(error="no-such-fact")
            t = time.time()
            ent = (entity or old["entity"]).strip()
            pred = (predicate or old["predicate"]).strip()
            val = (value if value is not None else old["value"]).strip()
            if not val:
                return dict(error="empty-value")
            obs_text = "【修正记忆】%s的%s是%s（原记：%s的%s是%s）" % (
                ent, pred, val, old["entity"], old["predicate"], old["value"])
            obs_id = self.dh.store.add_obs(t, "correct", "text", obs_text,
                                           source="用户", rel=1, seg_bnd=0.0)
            fid, _ = self.dh.store.write_fact(
                ent, pred, val, t, obs_id, rel=1, conf=0.95,
                force_new=True, revive_ok=True)
            changed_key = (ent != old["entity"] or pred != old["predicate"])
            if changed_key or self.dh.store.is_multi(pred):
                self.dh.store.retract_fact(fact_id)   # 修正=换掉这一条
            self.dh.store.commit()
            return dict(ok=True, fact_id=fid, changed_key=changed_key)

    def forget(self, fact_id):
        """撤回一条记忆（用户语义确认过：撤回不删经历，历史回放仍可还原）。"""
        with self.dh.store.lock:
            ok = self.dh.store.retract_fact(fact_id)
            if ok:
                self.dh.store.commit()
            return dict(ok=ok, retracted=bool(ok))

    # ------------------------------------------------ 配置
    def get_settings(self):
        return self.llm.masked()

    def set_settings(self, payload):
        """保存 API Key 等配置到本机文件（.gitignore 已排除，不进 git）。"""
        keep = {}
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, encoding="utf-8") as f:
                    keep = json.load(f)
            except (OSError, ValueError):
                keep = {}
        for k in ("api_key", "base_url", "model"):
            if k in payload:
                v = (payload[k] or "").strip()
                if v:
                    keep[k] = v
                else:
                    keep.pop(k, None)          # 空值 = 清除
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(keep, f, ensure_ascii=False, indent=1)
        return self.get_settings()

    def test_settings(self):
        return self.llm.test()


def make_handler(app, httpd=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):     # 安静些
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, name):
            path = os.path.join(STATIC, name)
            if not os.path.isfile(path):
                self._json(dict(error="not found"), 404)
                return
            ctype = {".html": "text/html; charset=utf-8",
                     ".css": "text/css; charset=utf-8",
                     ".js": "text/javascript; charset=utf-8"}.get(
                         os.path.splitext(name)[1], "application/octet-stream")
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # 本地应用，静态文件也不缓存——改了前端代码刷新即生效
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = self.path.split("?")[0]
            if route in ("/", "/index.html"):
                self._static("index.html")
            elif route == "/style.css":
                self._static("style.css")
            elif route == "/app.js":
                self._static("app.js")
            elif route == "/api/state":
                self._json(app.state())
            elif route == "/api/graph":
                self._json(app.display_graph())
            elif route == "/sea":
                self._static("sea.html")
            elif route == "/sea.js":
                self._static("sea.js")
            elif route == "/sea.css":
                self._static("sea.css")
            elif route == "/vendor/three.min.js":
                self._static("vendor/three.min.js")
            elif route == "/api/settings":
                self._json(app.get_settings())
            elif route == "/api/memories":
                self._json(app.memories())
            else:
                self._json(dict(error="not found"), 404)

        def do_POST(self):
            route = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self._json(dict(error="bad json"), 400)
                return
            if route == "/api/chat":
                text = (payload.get("text") or "").strip()
                if not text:
                    self._json(dict(error="empty"), 400)
                    return
                try:
                    self._json(app.on_chat(text))
                except Exception as exc:           # noqa: BLE001
                    self._json(dict(error="llm-failed", message=str(exc)[:300],
                                    reply=None), 502)
            elif route == "/api/settings":
                try:
                    self._json(app.set_settings(payload))
                except OSError as exc:
                    self._json(dict(error=str(exc)[:300]), 500)
            elif route == "/api/settings/test":
                self._json(app.test_settings())
            elif route == "/api/correct":
                try:
                    self._json(app.correct(int(payload.get("fact_id") or 0),
                                           entity=payload.get("entity"),
                                           predicate=payload.get("predicate"),
                                           value=payload.get("value")))
                except (ValueError, TypeError):
                    self._json(dict(error="bad fact_id"), 400)
            elif route == "/api/forget":
                try:
                    self._json(app.forget(int(payload.get("fact_id") or 0)))
                except (ValueError, TypeError):
                    self._json(dict(error="bad fact_id"), 400)
            elif route == "/api/shutdown":
                self._json(dict(ok=True))
                def _graceful():
                    if httpd is not None:
                        httpd.shutdown()
                        httpd.server_close()   # 释放端口
                    app.shutdown()
                    os._exit(0)                # 库已并、线程已停，干净退场
                                               # （不退会留僵尸：占着端口、库已关）
                threading.Thread(target=_graceful, daemon=True).start()
            elif route == "/api/associate":
                text = (payload.get("text") or "").strip()
                if not text:
                    text = app.last_user_text
                if not text:
                    self._json(dict(error="empty"), 400)
                    return
                try:
                    self._json(app.on_associate(text))
                except Exception as exc:           # noqa: BLE001
                    self._json(dict(error="associate-failed",
                                    message=str(exc)[:300]), 500)
            else:
                self._json(dict(error="not found"), 404)

    return Handler


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    # 第二个参数：空闲多少秒开始心智游移（默认 120，调试/演示可调小）
    idle = int(sys.argv[2]) if len(sys.argv) > 2 else None
    app = ChatServer(idle_seconds=idle)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    print("小星聊天页：http://127.0.0.1:%d   （Ctrl+C 退出）" % port)
    print("记忆库：data/xiaoxing.db | API Key 配置：%s（或 .env 的 ALIYUN_APIKEY）"
          % os.path.basename(SETTINGS_PATH))
    print("空闲 %s 秒后自动心智游移" % ("120" if idle is None else idle))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        app.dh.store.close()      # 优雅关闭：WAL 合并，重启零丢失


if __name__ == "__main__":
    main()
