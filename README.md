# 小星（virtual avatar）：带记忆的虚拟人

一个每 tick 做七件事的心智循环（感知→注意→检索→评估→反应→写回→离线），
底层是双时态 SQLite 记忆库 + PAD/OU 情绪动力学。它记得你说过的话、知道
自己从哪听来的、过期了会说「我记得以前是…现在不确定」，空闲时还会自己
回味（心智游移）。

架构见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 快速开始

```bash
pip install -r requirements.txt

# 网页版：聊天 + 识海 3D 记忆星图 + 记忆修正面板 + 语音条（模型自主发）
python web/server.py            # http://127.0.0.1:8787

# 离线演示：六个场景跑通全部记忆/情绪机制（不调任何 API）
python demo_memory_flow.py
```

网页右上角 ⚙ 配置 API Key（或 `.env` 写 `ALIYUN_APIKEY`）。**没有 Key 也能用**：
记忆点查、情绪、联想、修正全部照常工作，只是闲聊措辞换成离线兜底。

## 语音（CosyVoice3，可选）

发不发语音由小星自己决定——慢路径的 langgraph 工具循环里有一个 `send_voice`
工具，他判断此刻发语音更应景（比如刚说了句心里话、或你要求他说话）时就调用它，
回一条微信式语音条；没合适时机就正常打字。页面上没有「念出来」的按钮：
**语音是模型的选择，不是用户的开关**。

语音条是内存合成、base64 直达前端的，音频零落盘；「发过语音」这件事只以
一条摘要（`语音条 N″：内容`）写进记忆，所以刷新后语音条会退化成无音频的
摘要条，但小星记得自己说过什么。语音模型收在项目内 `third_party/`（已 gitignore）：

```bash
pip install -r requirements-voice.txt        # torch/transformers 等推理依赖
git clone https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
git -C third_party/CosyVoice submodule update --init --recursive
modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B \
    --local_dir third_party/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
```

设备自适应：cuda → mps（Apple 芯片 GPU）→ cpu，`VOICE_DEVICE` 可强制指定；
模型在后台线程预热，第一次真要发语音时通常只剩秒级合成。

语音终端版：`python main.py`（开口即说话，播放中可打断；摄像头视觉用
`VISION_*` 环境变量覆盖配置）。

## 它能做什么

- **记得住**：双时态事实（一行 = 一个值的一段有效期），时间旅行可回放「当时以为」
- **知道自己怎么知道的**：来源分级措辞——亲眼看到 / 你告诉过我 / 听X说 / 我猜
- **不编造**：LLM 抽取五道闸门，值必须逐字落地于原文；检索层没有的就弃权
- **会修正**：「不对，是…」当场改记忆并回报旧值；撤回不删经历，只封存
- **有情绪**：PAD 连续状态 + OU 动力学（惯性/回基线/噪声），四出口各读各的维
- **会联想**：激活扩散 + 联想循环，提一个人，他关联的人的记忆也亮
- **会走神**：空闲 120 秒开始心智游移，产出只能以「让我想起」的口吻说

## 目录

```
main.py               语音终端入口（输入泵/生成器/空闲三线程，说话中可随时打断）
demo_memory_flow.py   离线演示（六场景）
dh/                   心智系统本体（五层，见 ARCHITECTURE.md）
                      core/   七步循环、工作记忆、冲突/慢系统
                      memory/ 存储/写回/检索/激活图/值等价裁决
                      extract/正则 Tier 0 + LLM Tier 1
                      brain/  主图（大脑）：路由策略 + 节点 + 执行器
                      limbic/ 右脑：情绪状态机 + 七面调制 + 性格慢塑形
web/                  server.py（HTTP 薄层）+ chat_service.py（服务层）+ static/（前端）
voice/                CosyVoice 封装 + 情绪韵律后处理
vision/               摄像头 → 多模态模型描述
tests/                离线检验（python tests/test_*.py 直跑，不调任何 API）
data/                 记忆库（运行时生成，已 gitignore）
```
