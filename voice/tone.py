# -*- coding: utf-8 -*-
"""语气（tone）→ CosyVoice3 的 instruct 指令。

为什么需要这一层：outlet_speech 的 rate/pitch/energy 落到波形上只是
「整体变速 / 整体变调 / 整体增益」——实测 90% 的情绪动态只有
rate ±12%、pitch ±18%、energy ±25%，而且 pitch_shift 只抬 F0 中位、
F0 标准差几乎不动（只变花栗鼠/低沉，不产生语调起伏）。听感上那叫
「同一个语气稍微快慢高低」，不叫「换了个语气」。

真正的语气来自 CosyVoice3 的 instruct 通道：它同时改 F0 中位、
F0 标准差、响度、音节速率与停顿比例（同一条文本实测：
angry 309Hz/6.55 音节每秒 vs soft 167Hz/3.62 vs slow 102Hz/2.80），
这才是「同一句话说得不一样」。

两个硬约束（都是实测出来的，别改）：
  1. 指令文本必须严格按官方模板：`You are a helpful assistant. <指令><|endofprompt|>`
     （白名单见 third_party/CosyVoice/cosyvoice/utils/common.py:28 instruct_list）。
     不加模板头尾，指令会被当成正文朗读——实测指令字数 2/9/32/75 对应
     时长 3.92/5.64/13.72/14.4s，线性增长。
  2. 只能一条指令。多条叠加实测会退化成念稿，所以 tone 是单选，
     剩下的小幅调节才交给 rate/pitch/energy。
"""

# 官方 instruct_list 里与情绪/韵律有关的那几条（方言、佩奇风格不算情绪，不在此列）
TONE_LIB = {
    "neutral": None,
    "happy": "请非常开心地说一句话。",
    "sad": "请非常伤心地说一句话。",
    "angry": "请非常生气地说一句话。",
    "soft": "Please say a sentence in a very soft voice.",
    "loud": "Please say a sentence as loudly as possible.",
    "slow": "请用尽可能慢地语速说一句话。",
    "fast": "请用尽可能快地语速说一句话。",
    "robot": "你可以尝试用机器人的方式解答吗？",
}

# 语气已经决定了大幅度的韵律走向，后处理只留 ±5% 的微调余量：
# 再大就会把 instruct 好不容易做出来的韵律抹平，或者把音色拧坏
# （pitch_shift 到 1.22 时 spectral flatness ×3.07，听得出毛刺）。
TONE_PROSODY = {
    "neutral": (1.00, 1.00, 1.00),
    "happy":   (1.00, 1.04, 1.05),
    "sad":     (1.00, 0.97, 0.95),
    "angry":   (1.00, 1.03, 1.08),
    "soft":    (1.00, 0.98, 0.95),
    "loud":    (1.00, 1.00, 1.06),
    "slow":    (1.00, 1.00, 1.00),
    "fast":    (1.00, 1.00, 1.00),
    "robot":   (1.00, 1.00, 1.00),
}

TONE_LABEL = {
    "neutral": "平常",
    "happy": "开心",
    "sad": "低落",
    "angry": "生气",
    "soft": "轻声",
    "loud": "兴奋外放",
    "slow": "慢条斯理",
    "fast": "急促",
    "robot": "克制无机质",
}

_TEMPLATE = "You are a helpful assistant. {}<|endofprompt|>"


def normalize(name):
    """把外部传进来的语气名收敛到 TONE_LIB 里——模型瞎编、前端传空、
    大小写不一致都落到 neutral，不让它把合成打挂。"""
    if not name:
        return "neutral"
    key = str(name).strip().lower()
    return key if key in TONE_LIB else "neutral"


def instruct(name):
    """tone 名 → 可直接喂给 inference_instruct2 的指令文本；neutral 返回 None。"""
    body = TONE_LIB.get(normalize(name))
    return None if body is None else _TEMPLATE.format(body)


def spk_id(name):
    """tone → 预注册的音色槽名（tts_*.py 里一次性编码好，切换零成本）。"""
    return "tone_" + normalize(name)


def build_instruct(body):
    """给任意指令正文套官方模板（调试/扩展用）。"""
    return _TEMPLATE.format(body)


def pick_tone(pad, novelty=0.0):
    """PAD → tone 名。二维网格，单选。

    设计原则：p 定基调（好坏），a 定「这个基调有多外放」，d 只在
    负效价时区分「委屈(sad)」和「被冒犯(angry)」——因为 d 是支配感，
    高支配 + 负效价才是愤怒，低支配 + 负效价是难过，这是 OCC 的常识。

    阈值按 OU 过程实测的平稳分布定（a 轴 sd≈0.55，p 轴 sd≈0.56，
    ±0.30 大约切掉中间 45% 的日常状态，保证「平常」仍是多数）。
    """
    p, a, d = float(pad.p), float(pad.a), float(pad.d)
    if p <= -0.30:
        return "angry" if (a > 0.25 or d > 0.15) else "sad"
    if p >= 0.30:
        if a >= 0.50:
            return "loud"          # 高兴 + 高唤醒 = 外放的兴奋
        if a <= -0.30:
            return "soft"          # 高兴 + 低唤醒 = 满足的温柔
        return "happy"
    if a > 0.55:
        return "fast"              # 中性但躁
    if a < -0.45:
        return "slow"              # 中性但沉
    if a < -0.25:
        return "soft"
    return "neutral"


def prosody_for(name):
    """tone → (rate, pitch, energy) 微调三元组，给后处理用。"""
    return TONE_PROSODY.get(normalize(name), TONE_PROSODY["neutral"])


# tone 已经把基调定死之后，情绪三项只保留 1/4 的偏移量。
# 原因：instruct 好不容易做出的语调起伏，会被 pitch_shift 的整体变调
# 和增益压平盖掉——1.08 倍的音量配上 1.30 倍的增益，峰值直接撞 0.98 上限。
_DAMP = 0.25


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def combine(name, rate=1.0, pitch=1.0, energy=1.0):
    """情绪三项 × 语气微调 → 真正下给 TTS 的三项。

    tone=neutral 时原样透传（保持只有 PAD 驱动时的旧行为）；
    其它语气把情绪偏移压到 1/4，再叠加该语气的微调，并钳在安全区间里
    （rate 0.75~1.35 / pitch 0.90~1.12 / energy 0.85~1.20——超出去不是
    「更有情绪」，是花栗鼠和爆音）。"""
    name = normalize(name)
    r0, p0, e0 = prosody_for(name)
    if name == "neutral":
        return float(rate), float(pitch), float(energy)
    return (_clip((1.0 + (float(rate) - 1.0) * _DAMP) * r0, 0.75, 1.35),
            _clip((1.0 + (float(pitch) - 1.0) * _DAMP) * p0, 0.90, 1.12),
            _clip((1.0 + (float(energy) - 1.0) * _DAMP) * e0, 0.85, 1.20))


def mood_of(pad):
    """给前端/日志用的一行中文描述（不含数值）。"""
    return TONE_LABEL.get(pick_tone(pad), "平常")


def describe_for_prompt(pad=None):
    """给 system prompt 的一行提示：有哪些语气可用、当前建议哪个。

    模型看不到 PAD 数值，也不该看到——它只需要知道「此刻该用什么语气」，
    具体参数由宿主算。这一行只做两件事：告诉它 tone 参数合法值，
    以及给出按当前情绪推荐的默认值（模型可以不采纳）。"""
    names = "、".join("%s(%s)" % (k, TONE_LABEL[k]) for k in TONE_LIB)
    if pad is None:
        return "可用语气 tone：%s" % names
    cur = pick_tone(pad)
    return ("可用语气 tone：%s。按你此刻的状态，建议 tone=%s，"
            "但你可以按语境自己改。" % (names, cur))
