# -*- coding: utf-8 -*-
"""小星聊天页面：HTTP 薄层（路由 + 静态文件 + 启动）。

    GET  /api/state      小星此刻的情绪（PAD + 离散标签 + 表情 + 韵律）与自我叙事
    POST /api/chat       说话 -> 听见(hear) -> 记忆点查 or LLM -> 回复 + 情绪快照
                         （慢路径的 langgraph 工具循环里，模型自己决定是否调
                          send_voice 回一条微信式语音条：base64 内存直达前端，
                          本地零音频文件，经历照常入记忆）
    POST /api/speak      纯合成接口（文字 -> 语音条音频），不写记忆。
                         给模型侧/调试用；页面上没有「把上一句念出来」的按钮
    GET  /api/history    聊天记录（经历层重建；清屏分界时间之后的）
    POST /api/clear_chat 清空聊天窗口（记忆和识海不动，meta 记分界时间）
    GET  /api/graph      识海：记忆星图（three.js 3D 前端的数据源）
    GET  /api/memories   记忆面板：查看 / 修正 / 撤回小星记住的事实
    POST /api/correct    修正一条记忆（双时态：改值保留历史）
    POST /api/forget     撤回一条记忆（不删经历，历史回放仍可还原）
    GET/POST /api/settings          API Key 配置（只落本机，不进 git）
    POST /api/settings/test         用一条 1-token 请求验证连通
    POST /api/shutdown              优雅关闭（WAL 合并 + 线程退出）

编排逻辑在 chat_service.py（ChatServer），LLM 请求在 llm_client.py，
dh/ 是与网页无关的心智系统本体。运行：
    python web/server.py [端口=8787] [空闲游移秒数=120]
    浏览器打开 http://127.0.0.1:8787
"""
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from chat_service import ChatServer  # noqa: E402
except ImportError:     # 以包方式导入（import web.server）时走命名空间包路径
    from web.chat_service import ChatServer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

# 只接受 POST 的路由（/api/settings 例外，它 GET/POST 都有）。
# GET 打上来给 405 而不是 404——curl 不带 -d 默认发 GET，容易踩。
POST_ONLY = frozenset(("/api/chat", "/api/settings/test", "/api/correct",
                       "/api/forget", "/api/shutdown",
                       "/api/speak", "/api/clear_chat", "/api/ui"))


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
            ctype = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".jpg": "image/jpeg",
                ".png": "image/png"}.get(
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
            if route in POST_ONLY:
                self._json(dict(error="method not allowed",
                                hint="这个接口用 POST"), 405)
                return
            if route in ("/", "/index.html"):
                self._static("index.html")
            elif route == "/style.css":
                self._static("style.css")
            elif route == "/app.js":
                self._static("app.js")
            elif route == "/avatar.jpg":
                # 小星的头像图（ui_config.avatar_img 开关控制前端是否用）
                self._static("avatar.jpg")
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
            elif route == "/api/history":
                self._json(app.history_data())
            elif route.startswith("/img/"):
                # 用户发过的图片缩略图（项目内 data/images/），文件名白名单
                name = route[len("/img/"):]
                if not re.fullmatch(r"img_\d+\.(jpg|png)", name or ""):
                    self._json(dict(error="bad name"), 400)
                    return
                path = os.path.join(ROOT, "data", "images", name)
                if not os.path.isfile(path):
                    self._json(dict(error="not found"), 404)
                    return
                with open(path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg" if name.endswith("jpg") else "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif route == "/api/persona":
                from dh.limbic.persona import Persona
                # snapshot 会经 stage_info() 写 calib（阶段首现/晋升）——
                # 必须持锁，否则 HTTP 线程的写会混进聊天线程的在途事务
                with app.dh.store.lock:
                    self._json(Persona(app.dh).snapshot())
            else:
                # /api/ui 只收 POST（在 POST_ONLY 里，GET 会 405）；前端读
                # 外观配置走 /api/state 的 ui 段，这里不需要 GET 分支
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
                image = payload.get("image") or None
                if not text and not image:
                    self._json(dict(error="empty"), 400)
                    return
                try:
                    self._json(app.on_chat(text, image=image))
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
            elif route == "/api/speak":
                text = (payload.get("text") or "").strip()
                if not text:
                    self._json(dict(error="empty"), 400)
                    return
                try:
                    self._json(app.speak(text,
                                         float(payload.get("rate") or 1.0),
                                         float(payload.get("pitch") or 1.0),
                                         float(payload.get("energy") or 1.0),
                                         tone=(payload.get("tone") or "neutral")))
                except ValueError:
                    self._json(dict(error="bad params"), 400)
            elif route == "/api/clear_chat":
                # 只清窗口视图；经历/记忆/识海一律不动
                self._json(app.clear_chat())
            elif route == "/api/ui":
                # 小星（或用户）改前端外观：白名单校验+钳制在 set_ui 里
                self._json(app.set_ui(payload))
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
    print("记忆库：data/xiaoxing.db | API Key 配置：web/api_settings.local.json"
          "（或 .env 的 ALIYUN_APIKEY）")
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
