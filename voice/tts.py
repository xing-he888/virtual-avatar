"""
数字人语音合成 —— CosyVoice3 的最简封装

用法:
    from voice.tts import TTS

    tts = TTS()                                  # 加载模型(约13秒, 只做一次)
    tts.speak("你好呀", "hello.wav")              # 文字 -> wav 文件
"""
import os
import sys
import threading

# --- CosyVoice 是源码目录，不在 pip 里，得手动加进搜索路径 ---
COSYVOICE_DIR = r"C:\Users\xinghe\.workbuddy\tools\CosyVoice"
sys.path.insert(0, COSYVOICE_DIR)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS"))

import soundfile as sf                      # 读写 wav
from cosyvoice.cli.cosyvoice import AutoModel

from voice.prosody import apply_prosody     # 情绪韵律后处理（pitch/energy）
from voice import tone as tone_mod          # 语气 -> instruct 指令 + 微调

# --- 三个固定路径 ---
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
PROMPT_WAV = os.path.join(COSYVOICE_DIR, "asset", "zero_shot_prompt.wav")   # 音色参考音频

# CosyVoice3 要求带这个前缀，不能少
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def _register_tone_slots(auto):
    """把参考音频编码一次，派生每个语气一个音色槽。

    instruct2 的指令只改 prompt_text（进 LLM 侧），flow 侧要的「音色 +
    参考语音 token」对九种语气完全一样——所以只编码一次，再换 prompt_text
    复制出九份。切语气零编码成本。
    """
    fe = auto.frontend
    need = [tone_mod.spk_id(n) for n in tone_mod.TONE_LIB]
    if all(sid in fe.spk2info for sid in need):
        return

    if "default" not in fe.spk2info:
        auto.add_zero_shot_spk(PROMPT_TEXT, PROMPT_WAV, "default")

    proto = dict(fe.spk2info["default"])
    proto.pop("text", None)
    proto.pop("text_len", None)
    for name in tone_mod.TONE_LIB:
        sid = tone_mod.spk_id(name)
        if sid in fe.spk2info:
            continue
        info = dict(proto)
        body = tone_mod.instruct(name) or "You are a helpful assistant. <|endofprompt|>"
        tok, tok_len = fe._extract_text_token(body)
        info["prompt_text"], info["prompt_text_len"] = tok, tok_len
        fe.spk2info[sid] = info


class TTS:
    def __init__(self):
        print("正在加载模型...")
        self.model = AutoModel(model_dir=MODEL_DIR, fp16=True)   # fp16 加载更快，显存略省

        # 预注册音色槽：参考音频编码一次，九种语气共用同一份音色编码
        # （instruct 的指令只改 prompt_text，flow 侧音色不变）
        _register_tone_slots(self.model)

        self._interrupt = threading.Event()   # 打断标志：说话途中随时可以叫停
        print("模型就绪（语气槽 %d 个）" % len(tone_mod.TONE_LIB))

    def speak(self, text, out_path="output.wav", rate=1.0, pitch=1.0, energy=1.0,
              tone="neutral"):
        """把一句话合成语音，存成 wav，返回文件路径

        语气（tone）走 CosyVoice3 的 instruct 通道；rate/pitch/energy 是
        它上面 ±10% 的微调（见 voice/tone.combine）——不再是「整体变调 +
        整体增益」那种只能改音色的做法。
        """
        audio, sr = self.render(text, rate, pitch, energy, tone)
        sf.write(out_path, audio, sr)
        return out_path

    def render(self, text, rate=1.0, pitch=1.0, energy=1.0, tone="neutral"):
        """合成到内存：返回 (numpy 音频, 采样率)。上层据此直接 base64。"""
        audio = self._synth(text, rate, pitch, energy, tone)
        return audio, self.model.sample_rate

    def _synth(self, text, rate=1.0, pitch=1.0, energy=1.0, tone="neutral"):
        """整段生成再后处理——旧版在 for 循环里逐段 sf.write，
        模型分多段出音频时前面几段会被覆盖掉，只剩最后一段。"""
        import numpy as np
        name = tone_mod.normalize(tone)
        rate, pitch, energy = tone_mod.combine(name, rate, pitch, energy)
        chunks = []
        for piece in self._inference(text, name, rate):
            chunks.append(piece["tts_speech"].squeeze(0).cpu().numpy())
        audio = (np.concatenate(chunks).astype("float32") if len(chunks) > 1
                 else chunks[0].astype("float32"))
        return apply_prosody(audio, self.model.sample_rate,
                             pitch=pitch, energy=energy)

    def _inference(self, text, tone_name, speed):
        if tone_name == "neutral":
            return self.model.inference_zero_shot(
                text, "", "", zero_shot_spk_id="default",
                stream=False, speed=speed)
        return self.model.inference_instruct2(
            text, "", PROMPT_WAV, zero_shot_spk_id=tone_mod.spk_id(tone_name),
            stream=False, speed=speed)

    def play(self, text, rate=1.0, pitch=1.0, energy=1.0, tone="neutral"):
        """播放语音，途中可被 stop() 打断

        注意：先整段生成完再播，避免 RTF>1 导致的断断续续
        """
        import time
        import numpy as np
        import sounddevice as sd

        audio = self._synth(text, rate, pitch, energy, tone)

        # 补 300ms 静音：模型生成的音频结尾几乎没空白，蓝牙耳机的传输延迟会吃掉最后一个字
        sr = self.model.sample_rate
        padding = np.zeros(int(sr * 0.3), dtype="float32")
        audio = np.concatenate([audio, padding])

        dur = len(audio) / sr
        print(f"      [语气 {tone}，共 {dur:.2f} 秒音频]")

        self._interrupt.clear()
        sd.play(audio, sr)          # 非阻塞，方便随时叫停

        # 一直等到播完，或者被打断
        while True:
            stream = sd.get_stream()
            if stream is None or not stream.active:
                break
            if self._interrupt.is_set():
                sd.stop()
                print("      [被打断]")
                return
            time.sleep(0.05)

        time.sleep(0.2)   # 蓝牙缓冲，别让硬件吃掉尾音
        print("      [播放完成]")

    def stop(self):
        """立刻打断正在播的语音"""
        self._interrupt.set()


if __name__ == "__main__":
    tts = TTS()
    path = tts.speak("你好，我是你的虚拟人。今天想聊点什么？", "demo.wav")
    print("已生成:", path)
