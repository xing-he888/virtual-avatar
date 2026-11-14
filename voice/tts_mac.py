"""
数字人语音合成（Mac/便携版）—— CosyVoice3 的封装

与 voice/tts.py（Windows 版，保持原样不动）同接口：speak / play / stop。
平台分派见 voice/__init__.py 的 get_tts_class()——Windows 上永远加载不到本模块。

路径纪律：本文件不出现任何硬编码绝对路径——全部由项目根（__file__ 推导）
拼出，且每一项都能用环境变量覆盖，换机器零改动。

布局（全部收在项目内 third_party/，已 gitignore）:
    third_party/CosyVoice/                                          <- 源码（含 Matcha-TTS 子模块）
    third_party/CosyVoice/pretrained_models/<VOICE_MODEL_NAME>/     <- 模型权重

一次性获取（换机器重跑即可）:
    git clone https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
    git -C third_party/CosyVoice submodule update --init --recursive
    modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
        --local_dir third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B

环境变量（全部可选）:
    COSYVOICE_DIR      源码目录        （默认 <项目>/third_party/CosyVoice）
    VOICE_MODEL_NAME   模型子目录名    （默认 Fun-CosyVoice3-0.5B）
    VOICE_DEVICE       auto/cuda/mps/cpu（默认 auto：cuda -> mps -> cpu）
    VOICE_FP16         1 强开半精度    （默认仅 cuda 开——MPS/CPU 部分算子会糊）
"""
import os
import sys
import threading

# --- 项目根：由本文件位置推导，不做绝对路径假设 ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- CosyVoice 是源码目录，不在 pip 里 ---
COSYVOICE_DIR = os.environ.get(
    "COSYVOICE_DIR", os.path.join(_PROJECT_ROOT, "third_party", "CosyVoice"))
sys.path.insert(0, COSYVOICE_DIR)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS"))

import soundfile as sf                      # 读写 wav

from voice.prosody import apply_prosody     # 情绪韵律后处理（pitch/energy）
from voice import tone as tone_mod          # 语气 -> instruct 指令 + 微调

MODEL_DIR = os.path.join(
    COSYVOICE_DIR, "pretrained_models",
    os.environ.get("VOICE_MODEL_NAME", "Fun-CosyVoice3-0.5B"))
PROMPT_WAV = os.path.join(COSYVOICE_DIR, "asset", "zero_shot_prompt.wav")  # 音色参考音频

# CosyVoice3 要求带这个前缀，不能少（模型协议，不是路径）
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


def model_ready():
    """轻量探测：模型目录在不在。不触发 torch/cosyvoice 的重导入——
    /api/state 每 4 秒轮询一次，必须便宜。"""
    return os.path.isdir(MODEL_DIR)


def _patch_torchaudio_load():
    """torchaudio >= 2.9 把 .load 迁去了 TorchCodec（需要 ffmpeg 系统库）。
    CosyVoice 只用它读 wav——用 soundfile 提供等价实现
    （返回 (tensor[channels, frames], sr)，与 torchaudio 约定一致）。
    其余 resample/mel 都是纯张量算子，不受影响。"""
    import torch
    import torchaudio
    if getattr(torchaudio, "_via_soundfile_patch", False):
        return

    def load(filepath, *args, **kwargs):
        data, sr = sf.read(filepath, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = load
    torchaudio._via_soundfile_patch = True


def pick_device():
    """auto = cuda -> mps -> cpu。VOICE_DEVICE 可指定。"""
    import torch
    want = os.environ.get("VOICE_DEVICE", "auto").lower()
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return want


def _to_device(auto, device):
    """CosyVoice 内部设备探测只认 cuda/cpu（cosyvoice/cli/model.py:36），
    mps 在加载后手工把三个推理模块搬过去。"""
    import torch
    m = auto.model                      # CosyVoiceModel / CosyVoice2Model
    for name in ("llm", "flow", "hift"):
        getattr(m, name).to(device)
    m.device = torch.device(device)


class _MpsFloat64Guard:
    """CosyVoice 的 HiFi-GAN 推理会把 f0 预测器挪到 float64
    （hifigan/generator.py:716-717，CPU/CUDA 的精度小动作）——MPS 不支持
    双精度。调用期间把 torch.float64 暂时指到 float32，出上下文即恢复，
    不改第三方源码、不复制推理逻辑。"""

    def __enter__(self):
        import torch
        self._orig = torch.float64
        torch.float64 = torch.float32
        return self

    def __exit__(self, *exc):
        import torch
        torch.float64 = self._orig
        return False


def _guard_mps_float64(auto):
    """MPS 专用：给 hift.inference 套上 float64 守卫。"""
    import types
    hift = getattr(auto.model, "hift", None)
    if hift is None:
        return
    orig = hift.inference

    def inference(self, speech_feat, finalize=True):
        with _MpsFloat64Guard():
            return orig(speech_feat, finalize=finalize)

    hift.inference = types.MethodType(inference, hift)


def _register_tone_slots(auto):
    """把参考音频编码一次，派生每个语气一个音色槽。

    instruct2 的指令是在 prompt_text（进 LLM 侧）生效的，flow 侧要的
    「音色 + 参考语音 token」对九种语气完全一样——所以只编码一次，
    再把 prompt_text 换成各语气的指令 token 复制出九份。

    这样切语气零编码成本：实测每次合成不再有 wav 编码那约 2 秒；
    九槽注册总耗时 0.1 秒级（一次 frontend_zero_shot）。
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
    def __init__(self, device=None, fp16=None):
        print("正在加载模型...")
        _patch_torchaudio_load()
        from cosyvoice.cli.cosyvoice import AutoModel    # 重导入放惰性：轮询/导入本模块零开销

        self.device = device or pick_device()
        if fp16 is None:
            fp16 = os.environ.get("VOICE_FP16") == "1" or self.device == "cuda"
        self.model = AutoModel(model_dir=MODEL_DIR, fp16=fp16)
        if self.device == "mps":
            try:
                _to_device(self.model, "mps")
                _guard_mps_float64(self.model)
            except Exception as exc:         # noqa: BLE001
                print(f"      [MPS 不可用({exc})，退回 CPU]")
                self.device = "cpu"
                _to_device(self.model, "cpu")

        # 预注册音色槽：参考音频编码一次，之后每次合成直接复用（省约 2 秒/次）。
        # 每个语气一个槽——instruct2 的指令只在 prompt_text（LLM 侧）起作用，
        # 音色/flow 那半边完全相同，所以九个槽共用同一份参考音频编码。
        _register_tone_slots(self.model)

        self._interrupt = threading.Event()   # 打断标志：说话途中随时可以叫停
        print(f"模型就绪（设备={self.device} fp16={fp16} "
              f"语气槽 {len(tone_mod.TONE_LIB)} 个）")

    def speak(self, text, out_path="output.wav", rate=1.0, pitch=1.0, energy=1.0,
              tone="neutral"):
        """把一句话合成语音，存成 wav，返回文件路径

        语气（tone）走 CosyVoice3 的 instruct 通道——这才是「换语气」；
        rate/pitch/energy 只是它上面 ±10% 的微调（voice/tone.combine）。
        """
        audio, sr = self.render(text, rate, pitch, energy, tone)
        sf.write(out_path, audio, sr)
        return out_path

    def render(self, text, rate=1.0, pitch=1.0, energy=1.0, tone="neutral"):
        """合成到内存（不落盘）：返回 (numpy 音频, 采样率)。

        语音条场景由上层直接 base64 传给前端——音频不在本地留任何文件。"""
        audio = self._synth(text, rate, pitch, energy, tone)
        return audio, self.model.sample_rate

    def _synth(self, text, rate=1.0, pitch=1.0, energy=1.0, tone="neutral"):
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
        """按 tone 选音色槽：neutral 走原来的 zero_shot（音色最贴参考音频），
        其余走 instruct2 + 该语气的槽。instruct2 的 prompt_wav 参数在
        命中 spk2info 时不会被读，所以照样传 PROMPT_WAV。"""
        if tone_name == "neutral":
            return self.model.inference_zero_shot(
                text, "", "", zero_shot_spk_id="default",
                stream=False, speed=speed)
        return self.model.inference_instruct2(
            text, "", PROMPT_WAV, zero_shot_spk_id=tone_mod.spk_id(tone_name),
            stream=False, speed=speed)

    def play(self, text, rate=1.0, pitch=1.0, energy=1.0, tone="neutral"):
        """播放语音，途中可被 stop() 打断（先整段生成完再播，避免 RTF>1 断续）"""
        import time
        import numpy as np
        import sounddevice as sd

        audio = self._synth(text, rate, pitch, energy, tone)

        # 补 300ms 静音：模型生成的音频结尾几乎没空白，蓝牙耳机会吃掉最后一个字
        sr = self.model.sample_rate
        padding = np.zeros(int(sr * 0.3), dtype="float32")
        audio = np.concatenate([audio, padding])

        print(f"      [生成 {len(chunks)} 段, 共 {len(audio) / sr:.2f} 秒音频]")

        self._interrupt.clear()
        sd.play(audio, sr)          # 非阻塞，方便随时叫停

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
