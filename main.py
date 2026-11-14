"""
数字人主入口
    你说话 → dh 七步循环（记忆/情绪）→ 快路径点查 or 通义千问 → CosyVoice 合成语音

运行：
    C:\\Users\\xinghe\\miniconda3\\envs\\virtual_avatar\\python.exe main.py   （Windows）
    .venv/bin/python main.py                                                  （macOS，语音见 voice/tts_mac.py）

流程（三个线程）：
    输入泵线程  读一行 -> begin()：听见 + 立刻落盘（~200µs）-> 唤醒生成器
    生成器线程  run()：出锁算 LLM -> 回来验 epoch -> 作废 或 打印+说话
    空闲线程    静止 IDLE_WANDER_S=120s -> DH.idle()：巩固 + 心智游移

为什么不用同步图：同步 invoke 在算 LLM 的 3-10 秒里不读键盘，你打字也没人接，
「随时打断」在同步图里做不到。拆两个线程后，你说的话在按回车那刻就已落盘。

第三个线程是七步循环的第 7 步（网页端有、这里原来没有）：没人说话时它会
巩固（提炼自我叙事、塑形性格）和走神（在激活图上联想，以 rel=3 写回 obs）。
CLI 关窗即不存在，所以「空闲」只在你开着不动时才有意义。

与网页端共用同一份记忆库（data/xiaoxing.db）——网页里聊过的，这里都记得。
API Key 优先级：web/api_settings.local.json > .env 的 ALIYUN_APIKEY。
"""
import json
import logging
import os
import threading
import time

from dotenv import load_dotenv

from dh import DigitalHuman, MemoryStore, config
from dh.llm_bridge import LateExtractor
from voice import get_tts_class
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


# ---------- 记忆/情绪（七步循环，与网页端同一份库；路径钉死在项目内，
# 不随启动目录漂移——绝不写到项目外面去） ----------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DH = DigitalHuman(store=MemoryStore(os.path.join(_PROJECT_ROOT, "data", "xiaoxing.db")))

# 正则没吃下的句子 -> LLM 后台补记（不阻塞回复，五道闸门在 dh 里）
LATE = (LateExtractor(DH, lambda sys_p, usr: llm_chat(sys_p, usr, temperature=0.1))
        if _LLM_READY else None)

# 用户说到这些词，就抬头看一眼摄像头
VISION_WORDS = ["看", "瞧", "这是什么", "认得", "手上", "桌上", "画面", "屏幕"]


# ---------- 两个线程：输入泵 + 生成器 ----------
# 为什么不用 langgraph 的 invoke：invoke 是同步的，它算 LLM 那 3-10 秒里
# input() 根本不执行，你打字也没人接。要「随时打断」，就必须把「听」和
# 「想」拆成两次调用、放在两个线程里 —— 这是操作系统的活，不是图的活。
#
#     输入泵线程：读一行 -> begin()（听见+立刻落盘，~200µs）-> kick(epoch)
#     生成器线程：run(resume=st) -> 出锁算 LLM -> 回来验 epoch -> 作废或说话
#
# epoch 是一个只增不减的整数。每读进一句新话就 +1；生成器算完比对，
# 发现不是自己那一轮就直接丢掉（不写情绪、不写自己的话、不出声）。
# 它不取消 HTTP 请求（服务端照样在算），但用户感知上就是「被打断了」。
tts = get_tts_class()()   # 模型只加载一次，放全局；Windows/macOS 自动分派

_epoch = 0
_epoch_lock = threading.Lock()
_out_lock = threading.Lock()        # 保证打印/播放按顺序，不抢话
_kick = threading.Condition()       # 输入泵唤醒生成器的信号


def epoch_now():
    with _epoch_lock:
        return _epoch


def _bump_epoch():
    global _epoch
    with _epoch_lock:
        _epoch += 1
        return _epoch


def _play(reply, prosody):
    """语音丢子线程：主线程立刻回去读下一行，所以它说话时你能插话。"""
    if not reply:
        return
    p = prosody or {}
    threading.Thread(target=tts.play, args=(reply,),
                     kwargs=dict(rate=p.get("rate", 1.0),
                                 pitch=p.get("pitch", 1.0),
                                 energy=p.get("energy", 1.0),
                                 tone=p.get("tone", "neutral")),
                     daemon=True).start()


class Worker:
    """生成器线程。队列深度为 1：新话到了就顶掉旧话（说的话已经过时）。"""

    def __init__(self):
        self._pending = None            # 待处理的状态（只留最新一条）
        self._stop = False
        self._busy = False              # 空闲判断要用：正在算的时候不算「空闲」

    def busy(self):
        with _kick:
            return self._busy or self._pending is not None

    def kick(self, st):
        with _kick:
            self._pending = st
            _kick.notify()

    def stop(self):
        with _kick:
            self._stop = True
            _kick.notify()

    def loop(self):
        from dh.turn import run
        while True:
            with _kick:
                while self._pending is None and not self._stop:
                    _kick.wait()
                if self._stop:
                    return
                st = self._pending
                self._pending = None
                self._busy = True           # 从这刻起不算空闲
            # 出锁算：compose 里是 3-10 秒的网络等待，全程不持锁，
            # 所以输入泵随时能把新的一句话写进库
            try:
                out = run(DH, st["text"], llm_chat=llm_chat if _LLM_READY else None,
                          vision_desc=st.get("vision_desc"), epoch=st["epoch"],
                          epoch_now=epoch_now, resume=st)
            except Exception as e:      # noqa: BLE001  生成线程死了就再也
                # 没人回话（输入泵还在收字）——打日志继续，compose 内部的
                # LLM 异常兜底接不住的（如检索层崩溃）落到这里
                print(f"      [本轮失败] {type(e).__name__}: {e}")
                with _kick:
                    self._busy = False
                continue
            finally:
                with _kick:
                    self._busy = False
            if out.get("dropped"):
                # 算的过程中你又说话了 —— 这一轮的回复已经没意义，
                # 但你说的话在 begin() 里就落盘了，一个字都没丢
                continue
            if LATE is not None:
                LATE.submit(st["text"], st["t"], "用户", out["hear_out"])
            m = out["mood"]
            p = out["prosody"]
            with _out_lock:
                print(f"小星: {out['reply']}")
                print(f"      [情绪 P{m['p']:+.2f} A{m['a']:+.2f} D{m['d']:+.2f}"
                      f" | 语气{p.get('tone', 'neutral')} 语速{p['rate']:.2f}"
                      f" 音高{p['pitch']:.2f} 音量{p['energy']:.2f}]")
            _play(out["reply"], p)


_WORKER = Worker()


# ---------- 第二入口：空闲时自己巩固 + 走神 ----------
# 七步循环的第 7 步（DH.idle），网页端有、CLI 一直没有。
# 巩固 = 从反复出现的说法里提炼自我叙事 + 性格塑形；
# 游移 = 拿最近的事实当种子在激活图上联想 6 轮，以 rel=3 写回 obs。
#
# 和网页端 web/chat_service.py 的 _wander_loop 同构，差别是这里多两条：
#   1) 生成器正在算的时候不算空闲（否则它会一边回答问题一边走神）；
#   2) 走神产出打出来，让你看得见它在想什么。
_last_activity = [time.time()]
_idle_stop = threading.Event()


_last_clock = [time.time()]


def _idle_loop():
    """两件不同的事，同一个线程：

    1. **情绪时钟**（每 60 秒，2 µs）：没人说话时也让情绪继续往基线走。
       没有它，`tick()` 里 `affect.tick()` 在 `for item in attended:` 循环内，
       没有感知进注意就一步都不走——离开一小时再回来，情绪是在开口那一刻
       才跳一次，中间冻住。
    2. **第 7 步**（空闲 120 秒，~1 ms）：巩固 + 心智游移。
       CLI 关窗即不存在，所以这只在你放着不动时生效。
    """
    while not _idle_stop.wait(1.0):
        # ---- 情绪时钟：不管忙不忙都走，它是背景场
        if time.time() - _last_clock[0] >= config.EMO_CLOCK_S:
            try:
                with DH.store.lock:
                    DH.affect_clock()
                _last_clock[0] = time.time()
            except Exception as e:           # noqa: BLE001
                _last_clock[0] = time.time()
                print(f"      [情绪时钟失败] {e}")
        # ---- 第 7 步
        if time.time() - _last_activity[0] < config.IDLE_WANDER_S:
            continue
        if _WORKER.busy():                   # 正在回答，别插一脚
            _last_activity[0] = time.time()
            continue
        try:
            with DH.store.lock:
                res = DH.idle(t=time.time())
            DH.save_affect()                 # 游移/巩固也把情绪锚存档
        except Exception as e:               # noqa: BLE001
            _last_activity[0] = time.time()
            print(f"      [第7步失败] {e}")
            continue
        _last_activity[0] = time.time()      # 走神完重新计时
        w = res.get("wander") or {}
        narr = res.get("narratives") or 0
        if narr:
            print(f"      [巩固] 新提炼出 {narr} 条自我叙事")
        for th in (w.get("thoughts") or [])[:2]:
            print(f"      [自己想着] {th}")


# ---------- 输入泵（主线程） ----------
def main():
    from dh.turn import begin

    print("\n开始对话，输入 q 退出")
    print("（它说话时你直接打字回车就能打断——你说的话在按回车那刻就已落盘）")
    print(f"记忆库：data/xiaoxing.db | API Key："
          f"{'已配置（' + _LLM_CFG['model'] + '）' if _LLM_READY else '未配置，离线记忆模式'}\n")

    th = threading.Thread(target=_WORKER.loop, daemon=True)
    th.start()
    idle_th = threading.Thread(target=_idle_loop, daemon=True)
    idle_th.start()
    try:
        while True:
            try:
                text = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.lower() in ("q", "quit", "exit"):
                break

            _last_activity[0] = time.time()  # 有人说话就不算空闲
            tts.stop()                       # 上一句还没说完先掐掉
            vision_desc = None
            if any(w in text for w in VISION_WORDS):
                desc = look()                # 固定提问，别传用户原话
                vision_desc = desc
                print(f"      [看见] {desc[:50].replace(chr(10), ' ')}...")

            ep = _bump_epoch()               # 作废旧的那一轮
            # 只做「听见 + 落盘」：~200µs，不碰 LLM，所以能立刻回到 input()
            st = begin(DH, text, llm_chat=llm_chat if _LLM_READY else None,
                       vision_desc=vision_desc, epoch=ep)
            _WORKER.kick(st)
    finally:
        _idle_stop.set()
        _WORKER.stop()
        th.join(timeout=2.0)
        idle_th.join(timeout=2.0)
        DH.store.close()                     # 优雅关闭：合并 WAL


if __name__ == "__main__":
    main()
