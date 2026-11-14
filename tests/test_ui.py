# -*- coding: utf-8 -*-
"""前端 UI 控制层（/api/ui + 三个工具的白名单底座）的检验。
运行：.venv/bin/python tests/test_ui.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

PASS = 0


def ok(cond, name, extra=""):
    global PASS
    assert cond, "FAIL: %s %s" % (name, extra)
    PASS += 1
    print("  ✓ %s %s" % (name, extra))


tmp = tempfile.mkdtemp()
from web.chat_service import ChatServer                  # noqa: E402
app = ChatServer(store_path=os.path.join(tmp, "t.db"), idle_seconds=0)

# ---------------------------------------------------------------- 1 默认值
print("\n[1] 默认 UI 配置：预设解析成 CSS 变量")
cfg = app.ui_config()
ok(cfg["avatar_emoji"] == "🙂" and cfg["background"] == "deepspace", "出厂默认")
# 2026-11 二次改版契约（用户裁决：照 DeepSeek，纯白/纯黑）：每个预设是
# 一套完整 token 主题成套下发；默认「简白」对标 DeepSeek 浅色（白底+蓝
# 强调），「墨黑」对标深色；出厂头像色相为中性蓝 220。
ok(cfg["vars"]["--bg-a"] == "#FFFFFF" and cfg["vars"]["--accent"] == "#4D6BFE",
   "默认预设「简白」→ 纯白底 + DeepSeek 蓝",
   "%s / %s" % (cfg["vars"]["--bg-a"], cfg["vars"]["--accent"]))
ok(all(k in cfg["vars"] for k in ("--surface-top", "--bubble-user", "--text",
                                  "--dim", "--line", "--accent-ink", "--mask-bg")),
   "预设是完整主题（表面/气泡/文字/描边/遮罩成套）")
ok(cfg["avatar_hue"] == 220, "出厂头像光球色相 = 中性蓝 220")

# ---------------------------------------------------------------- 2 白名单
print("\n[2] 白名单校验：非法背景/超界数值被钳制或拒绝")
app.set_ui({"background": "my-custom-css"})              # 非法预设
ok(app.ui_config()["background"] == "deepspace", "非法预设名拒绝（LLM 注不进 CSS）")
app.set_ui({"avatar_hue": 999, "intensity": 7, "mood_glow": -3})
c = app.ui_config()
ok(c["avatar_hue"] == 360 and c["intensity"] == 1.0 and c["mood_glow"] == 0.0,
   "数值钳制 0-1 / 0-360", str((c["avatar_hue"], c["intensity"], c["mood_glow"])))
app.set_ui({"avatar_emoji": "✨✨✨✨✨✨✨✨✨九个字"})
ok(len(app.ui_config()["avatar_emoji"]) <= 8, "头像内容限长（防塞整段话）")
app.set_ui({"avatar_img": 5})
ok(app.ui_config()["avatar_img"] == 1, "avatar_img 收敛为开关（真值→1）")
app.set_ui({"avatar_img": 0})
ok(app.ui_config()["avatar_img"] == 0, "avatar_img 关→0（出厂默认 0=emoji）")

# ---------------------------------------------------------------- 3 真值持久
print("\n[3] 真值入库：重启（新实例同库）读到同一个人")
app.set_ui({"avatar_emoji": "🌙", "background": "neon"})
app2 = ChatServer(store_path=os.path.join(tmp, "t.db"), idle_seconds=0)
c2 = app2.ui_config()
ok(c2["avatar_emoji"] == "🌙" and c2["background"] == "neon",
   "UI 偏好跟库走（桌面/网页两客户端一致）")
ok("state" in dir(app) and app2.state()["ui"]["bg_name"] == "蔷薇",
   "/api/state 带 ui 段（前端轮询即得；neon 预设 2026-11 改版更名蔷薇）")
app2.shutdown()

# ---------------------------------------------------------------- 4 工具表
print("\n[4] LLM 工具表：三个 UI 工具已注册且 schema 收紧")
names = [t["function"]["name"] for t in ChatServer.VOICE_TOOLS]
for n in ("set_own_avatar", "set_chat_background", "set_presence"):
    ok(n in names, "工具已注册：%s" % n)
bg_tool = next(t for t in ChatServer.VOICE_TOOLS
               if t["function"]["name"] == "set_chat_background")
ok(bg_tool["function"]["parameters"]["properties"]["preset"]["enum"]
   == list(ChatServer.UI_BACKGROUNDS.keys()),
   "preset 是枚举（LLM 只能选白名单，不能自由发挥）")

# ---------------------------------------------------------------- 5 对话内自动联想
print("\n[5] recall_related：联想从按钮变成对话内自主行为")
app.set_ui({})                                # 无害调用，保持实例活跃
d = app.dh
d.store.set_owner("小明")
o = d.store.add_obs(1.0, "told", "text", "老王住在海淀区", source="用户")
d.store.write_fact("老王", "住址", "海淀区", 1.0, o, rel=1)
d.store.commit()
r = app.related_for("说起来老王最近怎么样")
ok("海淀区" in r and "老王" in r, "激活扩散捞出关联记忆（带来源标签）",
   (r or "")[:40].replace("\n", " / "))
names = [t["function"]["name"] for t in ChatServer.VOICE_TOOLS]
ok("recall_related" in names, "工具已注册（LLM 自主决定何时联想）")
ok(app.related_for("完全无关的话题量子涨落") == "",
   "无关联话题诚实返回空（不硬凑）")

app.shutdown()
print("\n全部通过：%d 项断言" % PASS)
