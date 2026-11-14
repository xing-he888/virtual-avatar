# -*- coding: utf-8 -*-
"""数字人的「看图」——用户在聊天里发来的图片 -> 客观文字描述。

与 camera.py（摄像头，Windows 版原样保留）同一族能力，但输入是
聊天里发来的图片（data URI），配置优先级一致：
    环境变量 VISION_APIKEY/VISION_BASE_URL/VISION_MODEL
    > web/api_settings.local.json（与网页端共用）
    > .env 的 ALIYUN_APIKEY
本文件零硬编码路径、零绝对路径。
"""
import json
import os
import base64

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "deepseek-v4.1-flash"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_QUESTION = ("这是用户在聊天里发给你的图片。客观描述你看到的内容："
                    "主要物体、人物、场景、可见的文字，以及整体氛围。")


def _look_cfg():
    cfg = {"api_key": None, "base_url": _DEFAULT_BASE_URL, "model": _DEFAULT_MODEL}
    try:
        with open(os.path.join(_ROOT, "web", "api_settings.local.json"),
                  encoding="utf-8") as f:
            saved = json.load(f)
        cfg.update({k: saved[k] for k in ("api_key", "base_url", "model")
                    if saved.get(k)})
    except (OSError, ValueError):
        pass
    for env_k, cfg_k in (("VISION_APIKEY", "api_key"),
                         ("VISION_BASE_URL", "base_url"),
                         ("VISION_MODEL", "model")):
        if os.environ.get(env_k):
            cfg[cfg_k] = os.environ[env_k]
    if not cfg["api_key"]:
        cfg["api_key"] = os.environ.get("ALIYUN_APIKEY")
    return cfg


def describe_image(image_uri, question=DEFAULT_QUESTION, timeout=90):
    """看一张图（data URI 或 http url），返回客观描述文本。

    模型不支持图片 / 没配 Key / 网络失败 -> 抛异常，由调用方决定怎么
    告诉用户（宁可明说看不见，不编造）。"""
    cfg = _look_cfg()
    if not cfg["api_key"]:
        raise RuntimeError("未配置 API Key，看不了图")
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"],
                    timeout=timeout)
    r = client.chat.completions.create(
        model=cfg["model"], max_tokens=4000,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": question},
        ]}])
    msg = r.choices[0].message
    out = (msg.content or "").strip()
    if not out:                       # 推理模型：正文空时回退思维链
        out = (getattr(msg, "reasoning_content", None) or "").strip()
    if not out:
        raise RuntimeError("模型没有返回描述")
    return out


def look_cfg_masked():
    """给前端的可用性提示（不回传 Key 全文）。"""
    cfg = _look_cfg()
    return dict(supported=bool(cfg["api_key"]), model=cfg["model"],
                base_url=cfg["base_url"])
