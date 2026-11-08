"""
数字人主入口
    你说话 → dh 七步循环（记忆/情绪）→ 快路径点查 or 通义千问 → CosyVoice 合成语音

运行：
    C:\\Users\\xinghe\\miniconda3\\envs\\virtual_avatar\\python.exe main.py

流程（langgraph 图）：
    START → think(记忆点查/大模型) → speak(带情绪韵律的语音) → END

与网页端共用同一份记忆库（data/xiaoxing.db）——网页里聊过的，这里都记得。
API Key 优先级：web/api_settings.local.json > .env 的 ALIYUN_APIKEY。
"""
import json
import logging
import os
import threading
import time
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from dh import DigitalHuman, MemoryStore
from dh.llm_bridge import LateExtractor, respond
from voice.tts import TTS
from vision.camera import look

# 关掉 openai/http 的调试日志，否则控制台全是请求细节
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()   # 读 .env，API Key 自动进环境变量

# ---------- 大模型 ----------
SETTINGS_PATH = "web/api_settings.local.json"     # 与网页端共用一份 Key 配置
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"


def _llm_cfg():
    """Key 优先级：本机配置（网页端设置页写的）> .env。"""
    cfg = dict(api_key=os.environ.get("ALIYUN_APIKEY"),
               base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL)
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update({k: saved[k] for k in ("api_key", "base_url", "model")
                        if saved.get(k)})
        except (OSError, ValueError):
            pass
    return cfg


_LLM_CFG = _llm_cfg()
_LLM_READY = bool(_LLM_CFG["api_key"])
if _LLM_READY:
    from openai import OpenAI
    llm = OpenAI(api_key=_LLM_CFG["api_key"], base_url=_LLM_CFG["base_url"],
                 timeout=120)


def llm_chat(system_prompt, user_text, temperature=None, max_tokens=16000):
    """max_tokens 给足——推理模型思维链吃 token，给少了正文为空。"""
    kw = dict(model=_LLM_CFG["model"],
              messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text}],
              max_tokens=max_tokens)
    if temperature is not None:
        kw["temperature"] = temperature
    resp = llm.chat.completions.create(**kw)
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:      # 正文空时回退思维链（抽取的 JSON 有时写在里面）
        content = (getattr(msg, "reasoning_content", None) or "").strip()
    return content


# ---------- 记忆/情绪（七步循环，与网页端同一份库） ----------
DH = DigitalHuman(store=MemoryStore("data/xiaoxing.db"))

# 正则没吃下的句子 -> LLM 后台补记（不阻塞回复，五道闸门在 dh 里）
LATE = (LateExtractor(DH, lambda sys_p, usr: llm_chat(sys_p, usr, temperature=0.1))
        if _LLM_READY else None)

# 用户说到这些词，就抬头看一眼摄像头
VISION_WORDS = ["看", "瞧", "这是什么", "认得", "手上", "桌上", "画面", "屏幕"]


# ---------- 图的共享状态 ----------
class State(TypedDict):
    user: str         # 用户说的话
    reply: str        # 虚拟人的回复文字
    prosody: dict     # 语速/音高/音量（情绪四出口的语音三项）
    mood: dict        # 当前 PAD 状态（p/a/d）
    audio: str        # 生成的语音文件路径


# ---------- 节点 1：想 ----------
def think(state: State) -> dict:
    """一轮完整 tick：听见→检索/评估→反应（respond 是 dh/llm_bridge 的桥）。"""
    user = state["user"]

    # 提到"看"就拍一张照片，把看到的内容一起交给模型
    vision_desc = None
    if any(word in user for word in VISION_WORDS):
        desc = look()          # 用固定提问，不要传用户原话，否则模型会以为是聊天附件
        vision_desc = desc
        print(f"      [看见] {desc[:50].replace(chr(10), ' ')}...")

    out = respond(DH, user, llm_chat=llm_chat if _LLM_READY else None,
                  vision_desc=vision_desc)
    if LATE is not None:
        LATE.submit(user, time.time(), "用户", out["hear_out"])

    p = out["prosody"]
    m = out["mood"]
    print(f"      [情绪 P{m['p']:+.2f} A{m['a']:+.2f} D{m['d']:+.2f}"
          f" | 语速{p['rate']:.2f} 音高{p['pitch']:.2f} 音量{p['energy']:.2f}]")
    return {"reply": out["reply"], "prosody": p, "mood": m}


# ---------- 节点 2：说 ----------
def speak(state: State) -> dict:
    """播放丢到子线程，主线程立刻回到输入 —— 它说话时你随时能插话"""
    p = state["prosody"]
    threading.Thread(target=tts.play, args=(state["reply"],),
                     kwargs=dict(rate=p["rate"], pitch=p["pitch"],
                                 energy=p["energy"]),
                     daemon=True).start()
    return {"audio": "(播放中)"}


# ---------- 组装图 ----------
tts = TTS()          # 模型只加载一次，放全局

graph = StateGraph(State)
graph.add_node("think", think)
graph.add_node("speak", speak)
graph.add_edge(START, "think")
graph.add_edge("think", "speak")
graph.add_edge("speak", END)
app = graph.compile()


# ---------- 主循环 ----------
def main():
    print("\n开始对话，输入 q 退出（它说话时你直接打字，回车就能打断）")
    print(f"记忆库：data/xiaoxing.db | API Key："
          f"{'已配置（' + _LLM_CFG['model'] + '）' if _LLM_READY else '未配置，离线记忆模式'}\n")
    try:
        while True:
            text = input("你: ").strip()
            if not text:
                continue
            if text.lower() in ("q", "quit", "exit"):
                tts.stop()
                break

            tts.stop()          # 上一句还没说完就先打断，像真人一样
            result = app.invoke({"user": text})
            print(f"小星: {result['reply']}")
            print(f"      [语音: {result['audio']}]")
    finally:
        # 优雅关闭：WAL 合并（强杀会丢最后几秒的补记）
        DH.store.close()


if __name__ == "__main__":
    main()
