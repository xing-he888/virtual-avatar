# -*- coding: utf-8 -*-
"""小星网页端的服务层：把 DigitalHuman 单例挂到可复用的 Python 对象上。

职责：对话主流程（快路径点查 / 慢路径 LLM）、动态记忆修正、后台 LLM 补记、
空闲心智游移线程、联想（激活扩散）、识海真值投影、记忆修正/撤回面板、设置。
HTTP 编排在 server.py（薄层）；LLM 请求在 llm_client.py。

情绪呈现的三个调研依据（dh/IMPLEMENTATION.md 同款出处风格）：
    Valdez & Mehrabian 1994  亮度/饱和度驱动唤醒感知，色相主导愉悦响应
                             -> 效价映射色相（暖橙/冷蓝），唤醒映射辉光
    Emerald 2024 "golden zone" 情绪表达过度反而失真 -> 表达克制：状态条+微色变
    Pezenka 2024 / USF 2022  情绪表达提升拟人化与温暖 -> 回复节奏是文字聊天的韵律通道
"""
import json
import os
import re
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dh import DigitalHuman, MemoryStore  # noqa: E402
from dh import config  # noqa: E402
from dh.llm_bridge import (apply_appraisal, build_system_prompt,
                           llm_route_question, offline_reply,
                           parse_appraisal, register_owner_if_intro)  # noqa: E402
from dh.core.tick import looks_like_question  # noqa: E402
from dh.brain import policy as mind_policy   # noqa: E402  主图的路由策略（大脑）

import numpy as np  # noqa: E402

try:
    from llm_client import LLMClient, DEFAULT_BASE_URL, DEFAULT_MODEL  # noqa: E402
except ImportError:     # 以包方式导入（import web.chat_service）时走命名空间包路径
    from web.llm_client import LLMClient, DEFAULT_BASE_URL, DEFAULT_MODEL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(HERE, "api_settings.local.json")

# PAD 离散标签锚点（Mehrabian & Russell 1974 情绪集 / Russell 1980 环状模型，
# 坐标取 [-1,1] 近似值）。设计纪律：内部永远是向量，离散标签只在输出时生成。
PAD_ANCHORS = [
    ("喜悦", (0.8, 0.3, 0.3)), ("兴奋", (0.6, 0.8, 0.2)),
    ("满足", (0.7, -0.3, 0.4)), ("平静", (0.3, -0.5, 0.1)),
    ("自豪", (0.4, 0.4, 0.8)), ("无聊", (-0.3, -0.6, -0.2)),
    ("低落", (-0.7, -0.4, -0.3)), ("沮丧", (-0.8, -0.1, -0.4)),
    ("焦虑", (-0.5, 0.6, -0.7)), ("恼火", (-0.6, 0.7, -0.3)),
    ("紧张", (-0.3, 0.5, -0.6)), ("委屈", (-0.45, 0.15, -0.6)),
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
    sp = dh.affect.outlets()["speech"]
    return dict(p=round(p, 3), a=round(a, 3), d=round(d, 3),
                label=pad_label(p, a, d), emoji=pad_emoji(p, a),
                rate=sp["rate"], pitch=sp["pitch"], energy=sp["energy"],
                lang=dh.affect.outlets()["language"])


class ChatServer:
    """把 DigitalHuman 单例挂到 HTTP 上。store/settings 路径可注入（测试用）。

    idle_seconds > 0 时起一个后台线程：空闲够久就跑第 7 步
    （巩固 + 心智游移），游移产出经 /api/state 推给前端主动开口。
    """

    def __init__(self, store_path=None,
                 settings_path=SETTINGS_PATH, seed=0,
                 idle_seconds=None, extractor=None):
        # 库钉在项目内（由代码位置推导，不随启动目录漂移——不污染别处）
        if store_path is None:
            store_path = os.path.join(ROOT, "data", "xiaoxing.db")
        os.makedirs(os.path.dirname(os.path.abspath(store_path)) or ".", exist_ok=True)
        store = MemoryStore(store_path)
        self.dh = DigitalHuman(store=store, seed=seed)
        # 向量索引对齐：库里的 vec 表是真值，data/chroma 是派生物。
        # 进程被强杀 / 首次接入 / 换了嵌入模型时数量会对不上，这里补齐。
        # 放在构造里（异步）：重建 10 万条要几十秒，不能挡启动
        threading.Thread(target=self._sync_index, daemon=True).start()
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
        # 异步打断：epoch = 用户输入的轮次；LLM 在锁外飞时来了新话 -> 旧话标记「迟到」
        self._chat_epoch = 0
        self._inflight_epoch = None
        self._inflight_question = None
        # 语音：模型惰性加载（首次十几秒），失败记住原因不再重试
        self._tts = None
        self._tts_state = "untried"           # untried / ready / failed
        self._tts_error = None
        self._tts_lock = threading.Lock()
        self._speak_lock = threading.Lock()   # 合成同一时刻只跑一条（模型非线程安全）
        self._stop = threading.Event()
        if self.idle_seconds and self.idle_seconds > 0:
            self._th = threading.Thread(target=self._wander_loop, daemon=True)
            self._th.start()
        # 语音模型后台预热：模型自己决定发语音的那一下，等的是秒级合成，
        # 不是几十秒加载
        threading.Thread(target=self._get_tts, daemon=True).start()

    def stop(self):
        """停掉游移线程（测试清理用）。join 等它退出，别跟关库赛跑。"""
        self._stop.set()
        th = getattr(self, "_th", None)
        if th is not None and th.is_alive():
            th.join(timeout=2.0)

    def shutdown(self):
        """优雅关闭：停线程 + 存情绪状态 + 提交合并 WAL + 关库。"""
        self.stop()
        try:
            self.dh.save_affect()      # 情绪状态 + 墙钟锚（重启后继续衰减）
        except Exception:              # noqa: BLE001
            pass
        try:
            self.dh.store.close()      # close 内含 commit + TRUNCATE checkpoint
        except Exception:              # noqa: BLE001
            pass

    def _sync_index(self):
        """后台对齐向量索引。失败静默——检索会退回 numpy 暴力扫。"""
        try:
            with self.dh.store.lock:
                info = self.dh.store.sync_vecindex()
            if info:
                print("[vecindex] %s" % info, flush=True)
        except Exception as e:                      # noqa: BLE001
            print("[vecindex] 同步失败（检索降级为暴力扫）：%s" % e, flush=True)

    def _wander_loop(self):
        """两件不同的事，同一个线程：

        1. **情绪时钟**（每 60 秒，2 µs）：没人说话时也让情绪继续往基线走。
           `tick()` 的 affect 步进在 `for item in attended:` 循环内，没有感知
           进注意就一步不走——没这个时钟，服务开着但没人聊天时情绪是冻住的。
        2. **第 7 步**（空闲 idle_seconds，~1 ms）：巩固 + 心智游移。
        """
        last_clock = time.time()
        while not self._stop.is_set():
            time.sleep(1.0)
            if time.time() - last_clock >= config.EMO_CLOCK_S:
                try:
                    with self.dh.store.lock:
                        self.dh.affect_clock()
                    last_clock = time.time()
                except Exception:                  # noqa: BLE001
                    last_clock = time.time()
            if time.time() - self.last_activity < self.idle_seconds:
                continue
            try:
                with self.dh.store.lock:
                    res = self.dh.idle(t=time.time())
                self.dh.save_affect()          # 游移/巩固也把情绪锚存档
            except Exception:                          # noqa: BLE001
                self.last_activity = time.time()
                continue
            self.last_activity = time.time()           # 游移完重新计时
            w = res.get("wander") or {}
            # 被打断的话题：最新一条打断锚还没被「已补答/已续上」盖掉 ->
            # 空闲+当前心情下主动提起来（虚拟人自己看着心情在未来提取）
            try:
                row = self.dh.store.con.execute(
                    "SELECT payload FROM obs WHERE kind='infer' AND channel='meta' "
                    "ORDER BY obs_id DESC LIMIT 1").fetchone()
                if row and "插了话" in row["payload"]:
                    w.setdefault("thoughts", []).insert(
                        0, row["payload"][:44] + "——要接着聊吗")
            except Exception:                          # noqa: BLE001
                pass
            if w.get("thoughts"):
                self.last_wander = dict(t=time.time(),
                                        thoughts=w["thoughts"][:3],
                                        touched=w.get("touched", 0))
                if w.get("seen"):
                    acc = {k: min(1.0, v / 3.0) for k, v in w["seen"].items()}
                    self.push_activation("游移", acc)

    # ------------------------------------------------ 对话主流程
    # ------------------------------------------------ 微信式语音条
    # 语音是模型手里的「工具」：在 langgraph 工具循环里由模型自己决定
    # 什么时候调用 send_voice（想听你说话/发条语音/或它自己判断应景），
    # 不是关键词匹配——用户不用说暗号。VOICE_REQ_RE 只在工具不可用时
    # （网关不支持 function calling）作为降级触发器。
    VOICE_REQ_RE = re.compile(
        r"(发[一条]?语音|语音条|语音说说|说给我听|说给wo听|读出来|念出来|用说的|说成语音|想听你说话|想听你说)")

    # 注册给模型的工具表（OpenAI function calling 格式）
    VOICE_TOOLS = [{
        "type": "function",
        "function": {
            "name": "send_voice",
            "description": "把一段话以微信语音条的形式发给用户（音频不落盘，点开即听）。"
                           "当用户想听你说话、明确要语音、或你判断此刻发语音更应景时调用；"
                           "content 写你要说的口语内容（一两句话），调用后不要再输出同样内容的文字。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string",
                                "description": "要说的口语内容"},
                    "tone": {
                        "type": "string",
                        "enum": ["neutral", "happy", "sad", "angry",
                                 "soft", "loud", "slow", "fast", "robot"],
                        "description": "用哪种语气说这条语音。不填就按你此刻的心情自动定；"
                                       "想表达反讽、冷淡、强撑平静这类「言不由衷」时，"
                                       "显式指定跟心情不一样的语气。",
                    },
                },
                "required": ["content"],
            },
        },
    }]
    VOICE_TOOL_NOTE = (
        "【语音工具】你有 send_voice 工具可以发微信语音条（音频不落盘）："
        "用户想听你说话、要求发语音、或你判断此刻发语音更应景（比如刚讲了个笑话、"
        "说了句心里话）时，调用它并把要说的口语放进 content；"
        "tone 可选，不填就按你此刻的心气自动定语气。"
        "调用过就不要再输出相同内容的文字。没有合适时机就不要调用。")

    def _agent_app(self):
        """langgraph 工具循环 agent（惰性编译一次；失败即永久降级为纯文本）。"""
        if getattr(self, "_agent", None) is not None:
            return self._agent
        if getattr(self, "_agent_failed", False):
            return None
        try:
            import operator
            from typing import Annotated, TypedDict
            from langgraph.graph import StateGraph, START, END

            class VState(TypedDict):
                messages: Annotated[list, operator.add]
                voice: dict
                pending: list
                tool_calls: list
                rounds: int

            def agent(state):
                resp = self.llm.chat_raw(state["messages"], tools=self.VOICE_TOOLS)
                msg = resp.choices[0].message
                tcs = getattr(msg, "tool_calls", None) or []
                d = {"role": "assistant",
                     "content": (msg.content or "").strip() or None}
                if tcs:
                    d["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in tcs]
                return {"messages": [d], "tool_calls": list(tcs),
                        "rounds": state.get("rounds", 0) + 1}

            def tools(state):
                """工具节点只登记「要发什么」，不在这一步合成——因为这一轮的
                <appraisal> 还没解析，此刻的 PAD 还是上一轮的情绪。合成放到
                agent 循环结束、apply_appraisal 之后（on_chat 里），语气才对得上。"""
                pend = list(state.get("pending") or [])
                outs = []
                for tc in state.get("tool_calls") or []:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except ValueError:
                        args = {}
                    if tc.function.name == "send_voice":
                        content = (args.get("content") or "").strip()
                        if not content:
                            outs.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": "content 为空，没发出去"})
                            continue
                        pend.append(dict(content=content,
                                         tone=(args.get("tone") or "").strip()))
                        outs.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "语音条已排入发送队列，这条消息之后发出"})
                    else:
                        outs.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "未知工具"})
                return {"messages": outs, "pending": pend, "tool_calls": None}

            def route(state):
                if state.get("tool_calls") and state.get("rounds", 0) < 3:
                    return "tools"
                return END

            g = StateGraph(VState)
            g.add_node("agent", agent)
            g.add_node("tools", tools)
            g.add_edge(START, "agent")
            g.add_conditional_edges("agent", route)
            g.add_edge("tools", "agent")
            app = g.compile()
            # 预检：网关必须真的支持 tools（1-token 探针），否则永久降级——
            # 不然第一条「发条语音」会静默变成纯文本
            self.llm.chat_raw([{"role": "user", "content": "ok"}],
                              tools=self.VOICE_TOOLS, max_tokens=8)
            self._agent = app
            return self._agent
        except Exception as exc:                   # noqa: BLE001
            self._agent_failed = True
            self._agent_error = str(exc)[:200]
            return None

    def _run_voice_agent(self, messages):
        """跑 langgraph 工具循环。返回 (最终文本, 待合成的语音请求列表)。

        语音不在循环里合成——send_voice 只登记 (content, tone)，真正的合成
        放在 caller 解析完 <appraisal>、PAD 更新之后，语气才对得上这一轮。

        网关不支持 tools / 图编译失败 / 循环异常 -> 纯文本回退（语音降级为
        关键词触发）。"""
        app = self._agent_app()
        if app is None:
            return ((self.llm.chat(messages[0]["content"],
                                   messages[-1]["content"]) or "").strip()), []
        state = {"messages": messages, "voice": None, "pending": [],
                 "tool_calls": None, "rounds": 0}
        try:
            out = app.invoke(state, config={"recursion_limit": 12})
        except Exception as exc:                   # noqa: BLE001
            self._agent_failed = True
            self._agent_error = str(exc)[:200]
            return ((self.llm.chat(messages[0]["content"],
                                   messages[-1]["content"]) or "").strip()), []
        final = ""
        for m in reversed(out.get("messages", [])):
            if m.get("role") == "assistant" and m.get("content"):
                final = m["content"]
                break
        return final, list(out.get("pending") or [])

    def _send_pending_voice(self, pending):
        """把 send_voice 排队的语音按当前 PAD 合成（appraisal 已落定之后调用）。

        tone 优先级：模型显式指定的 > 由当前 PAD 推的。模型显式指定时仍然
        用它，因为「嘴上说没事、声音在抖」这类反讽是它该有的表达自由。"""
        from voice import tone as tone_mod
        if not pending:
            return None
        item = pending[-1]                 # 一轮里只发最后一条，多余的丢弃
        content = item.get("content") or ""
        if not content:
            return None
        sp = self.dh.affect.outlets()["speech"]
        name = tone_mod.normalize(item.get("tone") or "")
        if name == "neutral" and not item.get("tone"):
            name = tone_mod.pick_tone(self.dh.affect.state)
        res = self.speak(content, rate=sp["rate"], pitch=sp["pitch"],
                         energy=sp["energy"], tone=name)
        if "voice" not in res:
            return None
        out = dict(res["voice"])
        out["tone"] = name
        out["content"] = content
        return out

    def _last_reply_text(self):
        with self.dh.store.lock:
            row = self.dh.store.con.execute(
                "SELECT payload FROM obs WHERE kind='told' AND channel='text' "
                "AND source='小星' ORDER BY obs_id DESC LIMIT 1").fetchone()
        return row["payload"] if row else None

    def _voice_reply(self, text, t, hear_out):
        """语音条回复：说的内容 = 句子去掉触发词的剩余话题（LLM 组织成口语，
        没配 Key 就直接念）；没话题就念小星最近一条回复。经历入记忆。"""
        topic = self.VOICE_REQ_RE.sub("", text)
        topic = re.sub(r"^[，,。！!？?\s]+|[，,。！!？?\s]+$", "", topic)
        content = None
        if topic and self.llm.configured():
            try:
                content = (self.llm.chat(
                    "把下面的内容用一两句口语说出来，像在发微信语音，不要任何格式：",
                    topic) or "").strip()
            except Exception:                      # noqa: BLE001
                content = None
        if not content:
            content = topic or self._last_reply_text() or "我还没说什么呢。"
        from voice import tone as tone_mod
        sp = self.dh.affect.outlets()["speech"]
        tone_name = tone_mod.pick_tone(self.dh.affect.state)
        res = self.speak(content, rate=sp["rate"], pitch=sp["pitch"],
                         energy=sp["energy"], tone=tone_name)
        now = time.time()
        with self.dh.store.lock:
            self._ensure_heard(self.dh.hear(content, speaker="小星", t=now),
                               content, "小星", now)
            snap = mood_snapshot(self.dh)
            if "voice" in res:
                # 「发过语音」作为经历写入记忆（只记摘要，音频零留存）
                self.dh.store.add_obs(now, "told", "voice",
                                      "语音条 %.0f″：%s" % (res["voice"]["seconds"],
                                                           content[:60]),
                                      source="小星", rel=1, seg_bnd=0.0)
                self.dh.store.commit()
        if "voice" not in res:
            rec2 = dict(role="xiaoxing", text="（语音不可用：%s）文字版：%s"
                        % (res.get("message", ""), content), mood=snap,
                        flags=[], sources=[], used_memory=True, stale=False,
                        cited=[], offline=not self.llm.configured(), late=False,
                        t=time.time())
        else:
            rec2 = dict(role="xiaoxing", text=content, mood=snap, flags=[],
                        sources=[], used_memory=True, stale=False, cited=[],
                        offline=not self.llm.configured(), late=False,
                        voice=res["voice"], t=time.time())
        self.history.extend([dict(role="user", text=text, t=0), rec2])
        return rec2

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
        # 主体一致性：修正句点名的实体若与锚定事实的主体不同，说明这句
        # 不是在否定锚（「不对，老王的电话是X」≠ 老王老婆的病情）——
        # hear 已按陈述句正常记下，这里放行走常规流，绝不错写锚定事实
        mentioned = {e[2] for e in self.dh.ex.find_entities(body)}
        if mentioned and fact["entity"] not in mentioned:
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

    def on_chat(self, text, image=None):
        """一轮对话。image 是用户发来的图片（data URI）——先看、作为
        rel=0 亲眼经历入记忆，再带着「看到了什么」进对话。

        纪律：store.lock 只锁记忆读写——LLM 在锁外飞，用户随时插话：
        插话立刻进入下一轮；被插断的那句答完会带 late 标记回来，
        打断现场写进经历层 obs（时光回溯的锚，可续、可被游移提起）。"""
        dh = self.dh
        image_file = None
        vision_desc = None
        vision_err = None
        if image:
            image_file, vision_desc, vision_err = self._see_image(image)
        with dh.store.lock:
            self.last_activity = time.time()
            self.last_user_text = text
            t = time.time()
            self._chat_epoch += 1
            my_epoch = self._chat_epoch
            # 插话检测：上一轮 LLM 还在飞 -> 打断现场入经历层
            if self._inflight_epoch is not None and self._inflight_question:
                dh.store.add_obs(
                    t, "infer", "meta",
                    "用户在「%s」还没答完时插了话：%s" % (
                        self._inflight_question[:24], text[:24]),
                    source="小星", rel=3, seg_bnd=0.0)
            # 1 它先听见：感知/注意/评估/写回（情绪与记忆在此更新）。
            # 会走 LLM 的话先听成「中性」——情绪改由 LLM 按性格的评价驱动
            # （慢路径里 apply_appraisal），词典只在离线/快路径兜底，避免双重计入。
            # 带图片的消息必须走慢路径（点查「看不见」图）
            will_llm = (self.llm.configured()
                        and (image_file is not None
                             or not self.VOICE_REQ_RE.search(text))
                        and not (dh.qrouter.route(text)
                                 and looks_like_question(text)))
            if text:
                hear_out = self._ensure_heard(
                    dh.hear(text, speaker="用户", t=t,
                            valence=0.0 if will_llm else None),
                    text, "用户", t)
            else:
                hear_out = {"obs_ids": [], "written": [],
                            "relation_facts": [], "relations_retracted": []}
            # 打断安全：用户的话在**听见这一刻**就落盘，不等 LLM 回来。
            # 原实现只在最后 :671 处 commit 一次，LLM 那 3-10 秒里如果用户
            # 又说话了、或者进程被杀，前面这句话就随事务一起没了。
            # commit 很便宜（WAL 一次 fsync 级，实测 ~200µs），值得。
            dh.store.commit()
            # 正则这次抽得可不可信（与 dh/brain/nodes 同一份判据）：
            # trusted 才跳过 LLM 补记，suspect/empty 都要交给 LLM 纠错
            from dh.brain.nodes import _trust
            hear_out["trust"] = _trust(text, dh.qrouter.route(text), dh)
            # 1.4 自报家门：登记属主——「老王是我的朋友」这类连线全靠它
            intro_ack = register_owner_if_intro(dh, text)
            # 1.5 动态记忆修正（用锁内的 hear_out 与 last_cited）
            corr = self._dynamic_correction(text, hear_out)
        if corr is None and intro_ack is not None:
            reply = intro_ack
            with dh.store.lock:
                self._ensure_heard(dh.hear(reply, speaker="小星", t=time.time() + 0.01),
                                   reply, "小星", time.time() + 0.01)
                snap = mood_snapshot(dh)
                dh.store.commit()
            rec2 = dict(role="xiaoxing", text=reply, mood=snap, flags=[],
                        sources=[], used_memory=True,
                        offline=not self.llm.configured(), stale=False, cited=[],
                        late=False, t=time.time())
            self.history.extend([dict(role="user", text=text, t=0), rec2])
            return rec2
        # 1.6 语音条请求的降级路径：langgraph 工具不可用（网关不支持
        # function calling）时，才退回关键词触发直接合成
        if self.VOICE_REQ_RE.search(text) and self._agent_app() is None:
            return self._voice_reply(text, t, hear_out)
        if corr is not None:
            reply = corr["reply"]
            with dh.store.lock:
                self._ensure_heard(dh.hear(reply, speaker="小星", t=time.time() + 0.01),
                                   reply, "小星", time.time() + 0.01)
                snap = mood_snapshot(dh)
                dh.store.commit()           # 修正即落盘
            self.last_cited = []
            rec2 = dict(role="xiaoxing", text=reply, mood=snap,
                        flags=[], sources=[], used_memory=True,
                        offline=not self.llm.configured(), stale=False,
                        cited=[], corrected=dict(fact_id=corr["fact_id"],
                                                 changed=corr["changed"]),
                        late=False, t=time.time())
            self.history.extend([dict(role="user", text=text, t=0), rec2])
            return rec2
        # 2 快路径：事实问句点查记忆，微秒级（锁内）。
        # parsed 来自问句路由器（属主消解 + 自学问法线索），不再靠手工小抄
        deferred = False
        fast = None
        learn_hint = None
        with dh.store.lock:
            parsed = dh.qrouter.route(text)
            if parsed is not None and looks_like_question(text):
                # record=False：问句的原话上面 _ensure_heard 已经留下底了，
                # 再让 ask 写一遍就会在同一秒落两条一样的 obs
                ans, phr, diag = dh.ask(text, record=False)
                if not ans.abstain:
                    srcs = [s for s in (ans.sources or []) if s not in ("用户", "我")]
                    fast = dict(reply=phr, flags=diag["conflict_flags"],
                                sources=srcs, stale=bool(ans.stale),
                                degraded=bool(ans.degraded), used_memory=True,
                                cited=list(ans.fact_ids))
        if fast is None and self.llm.configured() and parsed is None \
                and looks_like_question(text) \
                and ("我" in text or dh.ex.find_entities(text)):
            # 2.5 L2 问句解析兜底：L0/L1 没路由出来且句里有可查主体 → 问一次
            # 老师。解析合法即回填线索（自学习，落库在下面的锁内段），点查
            # 命中就直接答——省掉整趟 RAG。开放问句不问（解析必然失败）。
            # max_tokens 给足：推理模型的思维链吃 token，2000 会让预算花光在
            # reasoning 上、JSON 没生成（llm_client.py 的实测教训）
            parsed2, ans2 = llm_route_question(
                dh, lambda s, u: self.llm.chat(s, u, max_tokens=6000), text)
            if parsed2 is not None:
                learn_hint = parsed2["predicate"]
                if not ans2.abstain:
                    fast = dict(
                        reply=ans2.phrase(
                            directness=mind_policy.current_disposition(dh).directness),
                        flags=[], stale=bool(ans2.stale), used_memory=True,
                        sources=[s for s in (ans2.sources or [])
                                 if s not in ("用户", "我")],
                        cited=list(ans2.fact_ids))
        if fast is not None:
            reply, flags, srcs = fast["reply"], fast["flags"], fast["sources"]
            used_memory, stale = True, fast["stale"]
            self.last_cited = fast["cited"]   # 记下引用了哪条 -> 下一句可动态修正
            voice_payload = None
        elif self.llm.configured():
            # 3a 敷衍闸门（dh/brain/policy 的 defer 分支，网页端首次生效）：
            # 此刻说不动 + 这句话本身没内容（解析不出结构、认不出实体、短）
            # → 不花 3-10 秒调 LLM 硬凑，按人的样子应一声。CLI 侧早已有，
            # 这里补齐「情绪改控制流」在网页端的最后一处缺位。
            if mind_policy.should_defer(dh, text, parsed, True):
                reply = mind_policy.defer_line(dh)
                # used_memory=False：敷衍一声没引用记忆，别让前端挂「用了记忆」徽章
                flags, srcs, used_memory, stale = [], [], False, False
                voice_payload, deferred = None, True
                self.last_cited = []
            else:
                deferred = False
                # 3 慢路径：langgraph 工具循环——send_voice 等工具由模型自己
                # 决定调用与否（想听你说话/发条语音/它自己判断应景），音频零落盘。
                # 记忆上下文锁内取好，LLM 在锁外飞——插话随时进来
                resume = self._resume_context(text)
                with dh.store.lock:
                    prompt = build_system_prompt(dh, text)
                if resume:
                    prompt += resume
                if vision_desc:
                    prompt += ("\n【用户发来一张图片，这是你亲眼看到的画面】"
                               + vision_desc)
                elif vision_err:
                    prompt += "\n【用户发来一张图片，但你此刻没能看到】" + vision_err
                prompt += "\n" + self.VOICE_TOOL_NOTE
                self._inflight_epoch, self._inflight_question = my_epoch, text
                try:
                    messages = [{"role": "system", "content": prompt},
                                {"role": "user", "content": text}]
                    raw, pending = self._run_voice_agent(messages)
                finally:
                    self._inflight_epoch = self._inflight_question = None
                reply, apr = parse_appraisal(raw)
                # 情绪先落定（LLM 按性格的评价 -> PAD drive），快照才带上它
                apply_appraisal(dh, apr, text)
                # 语音在 appraisal 之后合成：这一轮的 PAD（进而语气）才算得准。
                # 旧顺序是先合成再 apply_appraisal，语音用的永远是上一轮的情绪。
                voice_payload = self._send_pending_voice(pending)
                flags, srcs, used_memory, stale = [], [], True, False
                self.last_cited = []
        else:
            voice_payload = None
            with dh.store.lock:
                # record=False：原话已由上面的 _ensure_heard 留底
                reply, meta = offline_reply(dh, text, record=False)
            flags, srcs, used_memory, stale = [], [], meta["used_memory"], False
            self.last_cited = []              # LLM 路径没有确切引用锚
        late = self._chat_epoch != my_epoch    # 答话期间用户又说了新话
        # 4 自己说的话也记 + 识海点亮 + 落盘（锁内）
        with dh.store.lock:
            now = time.time()
            hear_out_reply = None
            if reply:
                hear_out_reply = self._ensure_heard(
                    dh.hear(reply, speaker="小星", t=now + 0.01),
                    reply, "小星", now + 0.01)
            try:
                touched = [fid for out in (hear_out, hear_out_reply) if out
                           for fid in out.get("written", [])]
                if touched:
                    acc = dh.graph.diffuse({f: 1.0 for f in touched})
                    self.push_activation("对话", acc)
            except Exception:                          # noqa: BLE001
                pass
            if voice_payload:
                # 「发了语音条」作为经历入记忆（只记摘要，音频零留存）
                dh.store.add_obs(now, "told", "voice",
                                 "语音条 %.0f″：%s" % (voice_payload["seconds"],
                                                       (reply or text)[:60]),
                                 source="小星", rel=1, seg_bnd=0.0)
            if late:
                # 迟到的话也是补答：把打断锚标记成已了结（游移不再提它）
                dh.store.add_obs(now, "infer", "meta",
                                 "已补答：" + text[:30], source="小星",
                                 rel=3, seg_bnd=0.0)
            if learn_hint:
                # L2 学到的问法线索落库（与本轮写入同一 commit；学不成不挡回话）
                try:
                    dh.qrouter.learn(learn_hint, text, t=now)
                except Exception:              # noqa: BLE001
                    pass
            snap = mood_snapshot(dh)
            dh.save_affect()           # 情绪状态+墙钟锚持久化（隔夜衰减靠它）
            dh.store.commit()          # 消息即落盘（commit 后 PASSIVE checkpoint）
        offline = not self.llm.configured()
        rec2 = dict(role="xiaoxing", text=reply, mood=snap, flags=flags,
                    sources=srcs, used_memory=used_memory, offline=offline,
                    stale=stale, cited=list(self.last_cited), late=late,
                    deferred=deferred, voice=voice_payload, t=time.time())
        self.history.extend([dict(role="user", text=text, t=0), rec2])
        # LLM 补记：正则没抽出事实、但句子可能有信息 -> 后台抽取（不阻塞回复）
        # 判据与 web 之外一致：正则抽错时 written 也非空（`张姨说老王出院了`
        # 抽出 `张姨|病情|出院` 主语错人），所以看 trust 而不是 written 非空。
        if hear_out.get("trust") != "trusted" and (
                self.extractor is not None or self.llm.configured()):
            threading.Thread(target=self._late_extract,
                             args=(text, t, "用户", hear_out), daemon=True).start()
        return rec2

    def _resume_context(self, text):
        """时光回溯：用户说「继续/接着/刚才/回到」时，把最近一次被打断的
        话题从经历层取回来塞进提示词，并写下「已续上」的标记。"""
        if not any(w in text for w in ("继续", "接着", "刚才", "回到")):
            return ""
        with self.dh.store.lock:
            row = self.dh.store.con.execute(
                "SELECT payload FROM obs WHERE kind='infer' AND channel='meta' "
                "ORDER BY obs_id DESC LIMIT 1").fetchone()
            if row is None or "插了话" not in row["payload"]:
                return ""
            self.dh.store.add_obs(time.time(), "infer", "meta",
                                  "已续上：" + row["payload"][:40],
                                  source="小星", rel=3, seg_bnd=0.0)
        return "\n【之前被打断的话题】%s\n（用户说「%s」——自然地把这个话题接回去）" % (
            row["payload"], text[:20])

    def _see_image(self, image):
        """看用户发来的图：存缩略图（项目内 data/images/）+ 视觉模型描述 +
        rel=0 亲眼经历入记忆（他真的「记得」看过什么图）。
        返回 (文件名, 描述|None, 错误|None)。看不了就明说，不编造。"""
        import base64 as _b64
        from vision.look import describe_image
        try:
            head, _, b64 = image.partition(",")
            fmt = "jpg" if ("jpeg" in head or "jpg" in head) else "png"
            raw = _b64.b64decode(b64)
        except Exception as exc:               # noqa: BLE001
            return None, None, "图片解析失败：%s" % str(exc)[:80]
        name = "img_%d.%s" % (int(time.time() * 1000), fmt)
        img_dir = os.path.join(ROOT, "data", "images")
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(img_dir, name), "wb") as f:
            f.write(raw)
        try:
            desc = describe_image(image)
        except Exception as exc:               # noqa: BLE001
            desc, err = None, "我看不了图（%s）" % str(exc)[:100]
        else:
            err = None
        with self.dh.store.lock:
            ents = ",".join(dict.fromkeys(
                e[2] for e in self.dh.ex.find_entities(desc or ""))) or None
            self.dh.store.add_obs(time.time(), "perceive", "image",
                                  "[图片:%s] %s" % (name, desc or err or ""),
                                  source="用户", rel=0, seg_bnd=0.0,
                                  entity=ents)
        return name, desc, err

    def _ensure_heard(self, hear_out, text, speaker, t):
        """含问词（呢/？/吗…）的句子 tick 会跳过 writer（「问句不断言」——
        那是针对事实表的），但经历层的原话永远要留底：聊天记录重建、
        时光回溯、补记锚都靠它。hear 之后没 obs 就显式补一条。"""
        if not (hear_out.get("obs_ids") or []):
            with self.dh.store.lock:
                oid = self.dh.store.add_obs(t, "told", "text", text,
                                            source=speaker, rel=1, seg_bnd=0.0)
            hear_out["obs_ids"] = [oid]
        return hear_out

    def history_data(self):
        """聊天记录：直接从经历层 obs 重建（用户/小星的 told 发言、语音条、
        用户发的图片）。服务器重启也不丢——obs 本来就落盘。
        「清空聊天窗口」只裁这里的视图（meta 分界时间），记忆/识海不动。"""
        with self.dh.store.lock:
            cleared = self.dh.store.get_meta("chat_window_cleared_at")
            cleared = float(cleared) if cleared else None
            rows = self.dh.store.con.execute(
                "SELECT source, payload, t, channel FROM obs "
                "WHERE kind IN ('told','perceive') AND channel IN ('text','voice','image') "
                "AND source IN ('用户','小星') ORDER BY obs_id").fetchall()
        out = []
        for r in rows:
            if cleared is not None and r["t"] <= cleared:
                continue
            item = dict(role="user" if r["source"] == "用户" else "xiaoxing",
                        text=r["payload"], t=r["t"], channel=r["channel"])
            if r["channel"] == "image":
                m = re.match(r"\[图片:([^\]]+)\]\s*(.*)", r["payload"], re.S)
                if m:
                    item["image"] = m.group(1)
                    item["text"] = m.group(2)
            elif r["channel"] == "voice":
                # 经历层只存摘要（音频零留存）：语音条 3″：内容
                m = re.match(r"语音条\s*(\d+)\s*″[：:]\s*(.*)", r["payload"], re.S)
                if m:
                    item["seconds"] = int(m.group(1))
                    item["text"] = m.group(2)
            out.append(item)
        return dict(history=out)

    def clear_chat(self):
        """清空聊天窗口：经历/记忆/识海一律不动，只把窗口重建的分界时间
        写进 meta（持久，重启后依然只显示分界之后的对话）。"""
        with self.dh.store.lock:
            self.dh.store.set_meta("chat_window_cleared_at", repr(time.time()))
            self.dh.store.commit()
        return dict(ok=True)

    def _late_extract(self, text, t, speaker, hear_out):
        """LLM 补记：五道闸门 -> 写回 -> 识海点亮「回味」。失败静默（obs 永远在）。"""
        if not self._extract_busy.acquire(blocking=False):
            return                               # 上一条还没抽完，丢给下一条
        try:
            ex = self.extractor
            if ex is None and self.llm.configured():
                from dh.extract.llm import LLMExtractor, default_registry
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
                acc = (self.dh.graph.diffuse({f: 1.0 for f in out["fact_ids"]})
                       if out["fact_ids"] else {})
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
                # 同 graph_layer._derive_edges：Accelerate 虚假 FP 标志
                with np.errstate(all="ignore"):
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
                    tts=self.tts_state(),
                    n_facts=n_active,
                    n_obs=dh.store.obs_count())

    # ------------------------------------------------ 联想记忆
    def on_associate(self, text):
        """「提到X，还会想起什么」：激活扩散 + 联想循环（ρ/K/抽样）。

        种子 = activate_for_entities（GRAPH §4 第 3 条）：先按实体挑记忆、
        沿 关系.% 双向扩到关联人物，再取激活最强的几条做种子——
        提老陈，老王（他弟弟）的记忆也会跟着亮。
        """
        with self.dh.store.lock:
            dh = self.dh
            parsed = dh.ex.parse(text) or dh.ex.parse_question(text)
            ents = [e[2] for e in parsed["entities"]] if parsed else []
            if not ents:
                ents = [k for k in dh.ex.alias if k in text]
            if not ents:
                return dict(text="「%s」这个话题我还没记过什么，联想不起来。"
                                 "先跟我说说相关的事吧。" % text[:16],
                            kind="associate", touched=0, repeat_rate=0.0,
                            sources=[], used_memory=False, offline=True)
            acc = dh.graph.activate_for_entities(ents)
            if not acc:
                return dict(text="「%s」这个话题我还没记过什么，联想不起来。"
                                 "先跟我说说相关的事吧。" % text[:16],
                            kind="associate", touched=0, repeat_rate=0.0,
                            sources=[], used_memory=False, offline=True)
            seeds = dict(sorted(acc.items(), key=lambda kv: -kv[1])[:3])
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

    # ------------------------------------------------ 语音（CosyVoice）
    def tts_state(self):
        """/api/state 用的语音可用性快照。untried 时不加载模型，只做轻量探测。"""
        dev = getattr(self._tts, "device", None) if self._tts_state == "ready" else None
        supported = self._tts_state == "ready"
        if self._tts_state == "untried":
            from voice import tts_model_ready
            supported = tts_model_ready()
        return dict(state=self._tts_state, supported=supported,
                    device=dev, error=self._tts_error)

    def _get_tts(self):
        with self._tts_lock:
            if self._tts_state == "ready":
                return self._tts
            if self._tts_state == "failed":
                return None
            try:
                from voice import get_tts_class
                self._tts = get_tts_class()()   # Windows/macOS 自动分派
                self._tts_state = "ready"
                return self._tts
            except Exception as exc:                 # noqa: BLE001
                self._tts_error = str(exc)[:200]
                self._tts_state = "failed"
                return None

    def speak(self, text, rate=1.0, pitch=1.0, energy=1.0, tone="neutral"):
        """文字 -> 语音条：合成到内存、base64 直达前端——本地零音频文件。

        tone 是真正的语气通道（CosyVoice3 的 instruct，见 voice/tone.py）：
        rate/pitch/energy 只在它上面做 ±10% 微调。不传就是平常语气。

        纯合成，不碰记忆。经历层的那条 channel='voice' 摘要由调用方写：
        agent 的 send_voice 工具走 _run_voice_agent -> on_chat（chat_service
        第 596 行的 add_obs），降级路走 _voice_reply。语音该不该发由模型自己
        判断，前端不提供「把上一句念出来」的入口。
        首次调用会加载模型，HTTP 阻塞十几秒。"""
        tts = self._get_tts()
        if tts is None:
            return dict(error="tts-unavailable", message=self._tts_error or "")
        import io
        import base64
        import soundfile as _sf
        with self._speak_lock:
            if hasattr(tts, "render"):            # Mac/便携版：内存直达
                audio, sr = tts.render(text, rate, pitch, energy, tone)
                buf = io.BytesIO()
                _sf.write(buf, audio, sr, format="WAV")
                seconds = round(len(audio) / sr, 1)
            else:                                 # Windows 版：临时文件读完即删
                import tempfile
                fd, tmp = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    tts.speak(text, tmp, rate=rate, pitch=pitch, energy=energy,
                              tone=tone)
                    with open(tmp, "rb") as f:
                        raw = f.read()
                    info = _sf.info(tmp)
                    seconds = round(info.duration, 1)
                finally:
                    os.remove(tmp)
                buf = io.BytesIO(raw)
        uri = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
        return dict(voice=dict(data=uri, seconds=seconds, tone=tone),
                    state=self._tts_state, device=getattr(tts, "device", None))

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
