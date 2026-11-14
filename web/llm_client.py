# -*- coding: utf-8 -*-
"""LLM 客户端：DashScope OpenAI 兼容接口的薄封装。

Key 优先级：本机配置（web/api_settings.local.json，网页设置页写入）> .env 的 ALIYUN_APIKEY。
只做「照着配置发请求」一件事；对话编排、记忆注入在 chat_service.py。
"""
import json
import os

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"


class LLMClient:
    """OpenAI 兼容 chat 接口的薄封装。settings_path 可注入（测试用）。"""

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
        """给前端的打码视图：只有 Key 尾 4 位，绝不回传全文。"""
        key = self._cfg()["api_key"]
        if not key:
            return dict(has_key=False, tail="", base_url=self._cfg()["base_url"],
                        model=self._cfg()["model"])
        return dict(has_key=True, tail=key[-4:], base_url=self._cfg()["base_url"],
                    model=self._cfg()["model"])

    def chat_raw(self, messages, tools=None, timeout=60, temperature=None,
                 max_tokens=16000):
        """原生 messages + tools（function calling）接口，返回原始响应。

        langgraph agent 用它跑工具循环；普通 chat() 是它的单轮糖衣。"""
        cfg = self._cfg()
        if not cfg["api_key"]:
            raise RuntimeError("未配置 API Key")
        from openai import OpenAI
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"],
                        timeout=timeout)
        kw = dict(model=cfg["model"], messages=messages, max_tokens=max_tokens)
        if temperature is not None:
            kw["temperature"] = temperature
        if tools:
            kw["tools"] = tools
        return client.chat.completions.create(**kw)

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
        """一条 1-token 请求验证连通（设置页「测试连接」按钮）。"""
        try:
            out = self.chat("你是一个测试探针，只回答：ok", "ping", timeout=15)
            return dict(ok=True, sample=(out or "")[:40])
        except Exception as exc:                      # noqa: BLE001
            return dict(ok=False, error=str(exc)[:300])
