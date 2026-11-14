# 小星（virtual avatar）：带记忆的虚拟人

一个每 tick 做七件事的心智循环（感知→注意→检索→评估→反应→写回→离线），
底层是双时态 SQLite 记忆库 + PAD/OU 情绪动力学。它记得你说过的话、知道
自己从哪听来的、过期了会说「我记得以前是…现在不确定」，空闲时还会自己
回味（心智游移）。

他还有会慢慢长大的性格：被骂会变敏感、被教会变活泼（OCEAN 特质被经历
塑形，终身钳制不翻盘），信任随相处漂移，情绪基线跟着性格走、随昼夜节律
摆动；情绪和性格还各自加权在记忆的「存」与「取」上——强情绪情节忘得慢，
敏感的人忘不掉伤心事。

架构见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 快速开始

```bash
pip install -r requirements.txt

# 桌面版（Electron 壳，自己拉起 8790 后端）
cd desktop && npm install && npx electron .

# 网页版：聊天 + 识海 3D 记忆星图 + 成长面板 + 语音条（模型自主发）
python web/server.py            # http://127.0.0.1:8787

# 语音终端版：开口即说话，播放中可打断
python main.py

# 离线演示：六个场景跑通全部记忆/情绪/成长机制（不调任何 API）
python demo_memory_flow.py

# 全量检验：8 个套件直跑（不调任何 API）
.venv/bin/python tests/test_growth.py     # 其余 test_*.py 同款
```

右上角 ⚙ 配置 API Key（或 `.env` 写 `ALIYUN_APIKEY`）。**没有 Key 也能用**：
记忆点查、情绪、联想、修正全部照常工作，只是闲聊措辞换成离线兜底。

## 它能做什么

- **记得住**：双时态事实（一行 = 一个值的一段有效期），时间旅行可回放「当时以为」
- **知道自己怎么知道的**：来源分级措辞——亲眼看到 / 你告诉过我 / 听X说 / 我猜
- **不编造**：LLM 抽取五道闸门，值必须逐字落地于原文；检索层没有的就弃权
- **会修正**：「不对，是…」当场改记忆并回报旧值；撤回不删经历，只封存
- **有情绪**：PAD 连续状态 + OU 动力学（惯性/回基线/噪声），四出口各读各的维；
  基线由性格算出、随昼夜节律摆（凌晨低谷、傍晚最亮），跨重启按真实时间演化
- **有性格**：OCEAN 特质被经历塑形（有终身钳制）、信任漂移（破坏快建立慢）、
  四个发展阶段；性格与情绪共同决定话语分寸（terse/defer/elaborate 腿路）
- **记忆有加权**：存=编码强度（唤醒/惊讶/注意 × 尽责 C 增益），取=遗忘曲线
  （强情绪情节老化慢 × 神经质 N 只弯负性记忆的曲线）+ 情绪一致召回
- **有语气**：语音走 CosyVoice3 的 instruct 通道（不是整体变调的假语气），
  语气由情绪推、模型也可以按语境自己改
- **会联想**：激活扩散 + 联想循环，提一个人，他关联的人的记忆也亮
- **会走神**：空闲 120 秒开始心智游移，产出只能以「让我想起」的口吻说
- **看得见长大**：🌱 成长面板——特质条（出厂→现在）、情绪底色（含昼夜）、
  信任、阶段进度、自我叙事

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

## 目录

```
main.py               语音终端入口（输入泵/生成器/空闲三线程，说话中可随时打断）
demo_memory_flow.py   离线演示（六场景）
desktop/              Electron 桌面壳：spawn 后端(8790) → 开窗加载 localhost
dh/                   心智系统本体（六层，见 ARCHITECTURE.md）
                      core/   七步循环、工作记忆、冲突/慢系统、离线巩固
                      limbic/ 右脑：情绪状态机 + 昼夜节律 + 七面行为调制
                      growth/ 成长层：OCEAN 特质 + 发展阶段 + 塑形引擎 + 信任
                      memory/ 存储/写回/检索/激活图/值等价裁决/问句路由
                      extract/正则 Tier 0 + LLM Tier 1
                      brain/  主图（大脑）：路由策略 + 节点 + 执行器
web/                  server.py（HTTP 薄层）+ chat_service.py（服务层）+ static/（前端）
voice/                CosyVoice 封装 + 语气表（tone → instruct 指令）+ 情绪韵律后处理
vision/               摄像头 → 多模态模型描述
tests/                离线检验（.venv/bin/python tests/test_*.py 直跑，8 套件 269 项断言）
data/                 记忆库（运行时生成，已 gitignore）
```

## 已知保留项（裁决过、暂不实现）

- **情绪淹没 → 缄默腿**：被骂哭时对有内容的继续谩骂仍会回话（只有
  terse/defer 约束），缺「说不出话」
- **编码强度无倒 U**：情绪淹没时只有「记得更牢」，没有「脑子一片空白」

两处都在 ARCHITECTURE.md 有标注，想接随时说。
