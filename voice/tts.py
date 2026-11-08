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

# --- 三个固定路径 ---
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
PROMPT_WAV = os.path.join(COSYVOICE_DIR, "asset", "zero_shot_prompt.wav")   # 音色参考音频

# CosyVoice3 要求带这个前缀，不能少
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


class TTS:
    def __init__(self):
        print("正在加载模型...")
        self.model = AutoModel(model_dir=MODEL_DIR, fp16=True)   # fp16 加载更快，显存略省

        # 预注册音色：把参考音频编码一次存起来，之后不用每次重算（实测省约 2 秒）
        if "default" not in self.model.frontend.spk2info:
            self.model.add_zero_shot_spk(PROMPT_TEXT, PROMPT_WAV, "default")

        self._interrupt = threading.Event()   # 打断标志：说话途中随时可以叫停
        print("模型就绪")

    def speak(self, text, out_path="output.wav", rate=1.0, pitch=1.0, energy=1.0):
        """把一句话合成语音，存成 wav，返回文件路径

        rate/pitch/energy 是情绪四出口（dh/emotion.py outlet_speech）：
            rate  -> CosyVoice 的 speed（模型原生变速）
            pitch/energy -> 波形后处理（voice/prosody.py）
        """
        audio = None
        for piece in self.model.inference_zero_shot(text, "", "", zero_shot_spk_id="default",
                                                    stream=False, speed=rate):
            audio = piece["tts_speech"].squeeze(0).cpu().numpy()
            audio = apply_prosody(audio, self.model.sample_rate, pitch=pitch, energy=energy)
            sf.write(out_path, audio, self.model.sample_rate)
        return out_path

    def play(self, text, rate=1.0, pitch=1.0, energy=1.0):
        """播放语音，途中可被 stop() 打断

        注意：先整段生成完再播，避免 RTF>1 导致的断断续续
        """
        import time
        import numpy as np
        import sounddevice as sd

        chunks = []
        for piece in self.model.inference_zero_shot(text, "", "", zero_shot_spk_id="default",
                                                    stream=False, speed=rate):
            chunks.append(piece["tts_speech"].squeeze(0).cpu().numpy())

        audio = np.concatenate(chunks).astype("float32") if len(chunks) > 1 else chunks[0].astype("float32")
        audio = apply_prosody(audio, self.model.sample_rate, pitch=pitch, energy=energy)

        # 补 300ms 静音：模型生成的音频结尾几乎没空白，蓝牙耳机的传输延迟会吃掉最后一个字
        sr = self.model.sample_rate
        padding = np.zeros(int(sr * 0.3), dtype="float32")
        audio = np.concatenate([audio, padding])

        dur = len(audio) / sr
        print(f"      [生成 {len(chunks)} 段, 共 {dur:.2f} 秒音频]")

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
