# 小星 · 架构总览

> 小星不是「会说话的问答框」，是一个每 tick 做同样七件事、其中六件跟你的问题
> 无关的心智循环。这份文档是代码地图：每层做什么、谁依赖谁、数据怎么流。

## 全景图

```
        ┌─────────────── 入口层 ───────────────┐
        │  main.py（语音终端）   web/（网页）   demo_memory_flow.py（离线演示）
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │  服务层  web/chat_service.py          │  对话编排 / 动态修正 / 游移线程
        │         web/llm_client.py            │  LLM 请求（Key 配置、打码）
        │         web/server.py                │  HTTP 路由（薄层）
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │  心智层  dh/                          │
        │   core/    七步循环（下述 1-7 步）     │
        │   memory/  存储 / 写回 / 检索 / 激活图 │
        │   extract/ 正则 Tier 0 + LLM Tier 1   │
        │   brain/   主图（大脑）策略/节点/执行器 │
        │   limbic/  右脑  情绪状态机/调制面/塑形 │
        │   llm_bridge.py  记忆→LLM 的桥        │
        │   config.py      全部参数（带出处）    │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │  存储层  data/xiaoxing.db（SQLite+WAL）│  obs 追加锁死 / fact 双时态
        └──────────────────────────────────────┘
```

## dh/ 模块地图

| 层 | 模块 | 职责 |
|---|---|---|
| **core/** | `tick.py` | 七步循环：感知→注意→检索→评估→反应→写回→离线。`DigitalHuman.hear()/ask()/idle()` 是全部对外入口 |
| | `working_mem.py` | 工作记忆，4 槽硬上限，挤掉显著性最低的 |
| | `conflict.py` | 六信号冲突检测器 + 慢系统（显式 token 预算，默认不启动） |
| | `offline.py` | 第 7 步：巩固（自我叙事）+ 心智游移（联想写回 rel=3） |
| **limbic/**（右脑） | `emotion.py` | 情绪状态机：PAD + OCC/顺序评价 + OU 精确离散化 + 四出口。LLM 只指认评价维度（<appraisal>），数学负责结论。基线 mu = 性格 base_mu + 昼夜偏移（每步 refresh_mu）；长间隔回归贴「终点时刻」的基线。不变量「一句话至多一个效价来源」：hear 词典效价≠0 时本句的 apply_appraisal 整体跳过（CLI 离线/已解析问句不再双计，web/CLI 同判据） |
| | `rhythm.py` | 昼夜节律：mu(t) = 基线 + amp·cos(2π(h−peak)/24)，24h 周期、有界、极值可查表（谷在凌晨、峰在午后到傍晚）；基线热更新走 AffectSystem.set_base_mu（塑形后立即生效，不等重启） |
| | `disposition.py` | 右脑→大脑的调制器：mood/focus/talkativeness/directness/patience/breadth 七面进控制流（gate=情节编码门已接 obs.salience）；大脑只从这里拿情绪。**已知保留项（用户裁决 2026-11-14，暂不实现）**：①情绪淹没→缄默腿——骂哭态对有内容的继续谩骂仍走回话腿（只有 defer/terse 约束），缺「说不出话」；②编码强度的唤醒项单调无倒 U——只有「记得更牢」没有「淹没时脑子空白」。两处设计已在会话中评估过，等用户点头再接 |
| | `persona.py` | 兼容 shim → dh/growth（性格系统已升级为成长层） |
| **growth/**（成长层） | `traits.py` | OCEAN 五维特质真值（calib 只插不改）→ Gebhard/ALMA 公式 → PAD 基线喂 OU——情绪底色由特质算出 |
| | `stages.py` | 发展阶段（初生→幼生→成长→成熟，只进不退）：解锁=轮数+天数+话题多样性；调制学习率/游移/话长/信任建立（trust_mult，只放大正面增量） |
| | `persona.py` | 塑形引擎：归因规则表 + salience 加权 + EMA 微移 + 漂移钳制（出厂±0.3）+ 里程碑自我叙事；三个塑造接口（自主 shape / 对话式教养 nurture / 编程 shape_direct）；信任随经历漂移（非对称：破坏快建立慢，独立水位线单条结账）；恢复力按**经过时间**指数结算（τ=585 天：挂机一天≈0.17%、整月≈5%——旧「每次巩固回拉 1%」被空闲线程 120s 一次的节奏在一天内全额回滚）；特质写入时间戳只进不退；证据窗口带上界（未来行不跨巩固重复计费）；行为面系数放大到满漂移跨话语模式阈值（旧系数漂满也感知不到）；提示词「性格现状」把漂移译成第一人称祈使句（漂多少说多少）；前端 🌱 成长面板消费 /api/persona |
| **memory/** | `store.py` | SQLite 五区：obs（追加锁死）/ fact（双时态）/ event-mention-slot / vec / calib |
| | `writer.py` | 写回步：来源分级、传旧话不复活、关系互斥撤回、代词防护 |
| | `retrieve.py` | 三条检索路：点查（微秒）/ 语义 top-k / 激活扩散；幻觉治理措辞在这里落地。情绪与性格都加权在「存取」上：存=编码强度 gate（唤醒/惊讶/注意 × C 尽责增益），取=情节池遗忘减速（强编码老化慢 × N 神经质只弯负性记忆的曲线）+ 情绪一致召回（mood×valence） |
| | `kernel.py` | 遗忘的时间核：对数正态（50,465 条真实消息拟合，推翻指数） |
| | `graph_layer.py` | 激活图：share/link/sim 三类边、3 跳扩散、联想循环（ρ/K/抽样） |
| | `embed.py` | 文本→向量：bge-small-zh-v1.5（CLS pooling，512 维），缺模型降级回 hash256 |
| | `vecindex.py` | 向量索引：Chroma/HNSW，**派生**。SQLite 的 vec 表是唯一真值，整目录可删 |
| | `scoring.py` | 三路混合打分：候选集内 z 标准化 + 两条断言 |
| | `calib.py` | 遗忘常数标定：phi/back 回归 → 标定即冻结 |
| | `qroute.py` | 问句路由器（semantic router）：L0 属主消解/线索子串 + L1 问句嵌入 vs 自学线索 ≥0.6 → 点查；L2 LLM 解析兜底在 llm_bridge，解析成功回填线索——表自生长，零人工 |
| | `equiv.py` | 写入侧值等价裁决：词典归一 → 别名表 → 豁免（数字/关系）→ 前缀否决 → value-only 余弦（0.85）；confirm_equiv 不动 valid_from |
| **extract/** | `rules.py` | Tier 0 抽取：正则 + 别名表 + 关系检测（零成本确定性）；`normalize_value` 是 Tier 0/1 共用的值归一口 |
| | `llm.py` | Tier 1 抽取：五道闸门（schema/时态/值落地/谓词白名单/置信），宁可漏记不可编造 |
| **brain/** | `policy.py` | 主图的路由策略（大脑）：ack/defer/fast/offline/llm 五条腿，分支条件全部读记忆/心智状态，纯函数可单测 |
| | `nodes.py` | 主图节点实现（原 turn.py）：plan 调 policy.decide 定路由，compose 出锁执行五条腿 |
| | `graph.py` | 主图执行器：节点表/锁边界/epoch 协议/begin-run 拆分；`describe()` 可打印 |
| 平级 | `config.py` | 全部工程参数，每个数标出处（[SUMMARY §x] 等研究文档锚点） |
| | `context.py` | 检索上下文的唯一收集点：一次收集（A 精确/B 语义/C 图）多处消费，与排版分离 |
| | `llm_bridge.py` | 记忆上下文 → 系统提示词；情绪 → 语气指令；`respond()` 薄封装指向 dh/brain；`LateExtractor` 后台补记 |
| | `turn.py` | 兼容 shim：begin/run/_trust 原样可用，实现已搬进 dh/brain/ |
| | `world.py` | 合成 30 天对话世界（1800 轮 + 金标），评测协议载体 |

## 一次对话的数据流（web 版）

```
用户说话
  └→ ChatServer.on_chat()
       ├→ dh.hear(text)            # 1-4 步：感知/注意/检索/评估（情绪更新）
       ├→ 动态修正检查              # 「不对，其实是…」→ 当场改记忆 + 确认回复
       ├→ 快路径：parse_question    # 事实问句 → SQLite 点查，微秒级，0 token
       ├→ 慢路径：llm_bridge.build_system_prompt
       │     记忆上下文（来源分级措辞）+ 情绪语气指令 → LLMClient.chat
       ├→ dh.hear(reply, speaker="小星")   # 6 步：自己说的话也记
       ├→ store.commit()           # 消息即落盘（WAL + PASSIVE checkpoint）
       └→ 后台 LateExtractor       # 正则没吃下的句子 → LLM 五道闸门 → 补记

🔊 发语音（由模型自己决定，前端没有按钮）：
  慢路径 langgraph 工具循环 → 模型调 send_voice{content}
       └→ ChatServer.speak() → voice/__init__.py get_tts_class()
            → voice/tts_mac.py（非 Windows）或 voice/tts.py（Windows）
            → 内存合成 → base64 data:URI 直达前端（音频零落盘）
       └→ on_chat 写一条 obs(channel='voice')，只存「语音条 N″：内容摘要」
  降级：网关不支持 function calling 时，VOICE_REQ_RE 命中才走 _voice_reply
```

发语音是**模型的决定**，不是用户的遥控器：页面上没有「把上一句念出来」的
按钮，用户正常说话即可，模型判断此刻发语音更应景时才调用 send_voice。
前端只负责播放语音条（点开听，刷新后退化成无音频的摘要条）。

## 前端两条硬约定（踩过坑，别回退）

1. **作者写 `display` 的元素，必须自带 `[hidden]` 规则**。UA 的
   `[hidden] { display: none }` 优先级低于任何作者样式，所以
   `.mask`、`.pic-preview` 这类写了 `display: flex/inline-block` 的盒子，
   `el.hidden = true` 完全不生效。项目里的写法：
   `.mask[hidden] { display: none; }`、`.pic-preview[hidden] { display: none; }`。
2. **`.composer` 是纵向 flex**。图片预览（`#picPreview`）与输入行
   （`.composer-row`）是兄弟节点，靠 `flex-direction: column` 上下堆叠；
   漏掉它就是左右并排，预览跑到输入框左边。清预览时用
   `img.removeAttribute("src")`，不要写 `img.src = ""`（Safari 会把空串
   当相对 URL 解析回本页，画破图）。

## 设计纪律（改代码前先读）

1. **obs 追加锁死**（触发器物理禁止改/删）——改记忆是加新版本，不是回头改旧记录。
2. **fact 双时态**：一行 = 一个值的一段有效期；「现在」只是 `valid_to IS NULL` 的特例。
3. **事实记忆一律全写**，唤醒度门控只管情节记忆（E3/E5/E7 三次实验否证）。
   门控的落点是**编码强度**：写入时 gate(A,惊讶,注意) 存进 obs.salience，
   读出侧三根杠杆消费——想起更快（EMO_SAL_GAIN）/ 忘得更慢（SAL_AGE_SCALE，
   闪光灯记忆效应）/ 塑形更重（persona.shape 加权）+ 游移偏爱（offline 种子）。
   不写、不删任何经历行。
4. **来源分级 L0-L4 → 措辞**（我看到 / 你告诉过我 / 听X说 / 我猜）；过期 → 降级措辞；弃权放最后。
5. **LLM 只在最后一步出现**——点查/语义/扩散都是数学；LLM 抽取值必须逐字落地于原文。
6. **同种子同结果**：连情绪噪声都进 seed；昼夜节律的时刻进可注入的
   `affect.clock`（测试传固定 epoch）；确定性哈希嵌入（不用内置 hash()）。
7. 每个参数在 `config.py` 里标出处；被真实数据推翻的参数不在代码里。
8. **一句问话只落一条 obs**：问句被 tick 跳过写回（问句不断言），原话由
   `_ensure_heard()` 补底；此后调用 `dh.ask(text, record=False)`，
   否则同一秒会写两条一模一样的经历。
9. **索引是派生物，真值只有一个**：`data/xiaoxing.db` 的 vec 表存向量，
   `data/chroma/` 只是它的 HNSW 索引。索引删了/坏了/没装 chromadb，
   检索自动退回 numpy 暴力扫（`semantic_topk` 里的兜底），库一行不丢。
   换嵌入模型 = 换 collection 名，两种模型的余弦绝不混算。
10. **视图跟着库走**：`data/persona.md` 由 `Persona.md_path()` 推导——
    内存库（demo/测试）`md_path() is None`、不写文件；落盘库写在
    `<库同目录>/persona.md`。绝不用空 calib 去覆盖真实库的快照。
11. **写入侧值等价三段裁决**（dh/memory/equiv.py）：语义相同 ≠ 字面相同——
    词典归一（确定性）→ 别名表（历史裁决）→ value-only 余弦（开放集，0.85），
    变更信号词（搬到/换号…）一票否决，数字/关系谓词豁免，前缀包含且差
    ≥2 字确定性否决。confirm_equiv 只动 confirmed_t：**不加行、不 supersede、
    不改 valid_from、不改写 value**（先到为规范值，新表面形式进 value_alias）。
    字面完全相同的复述不进裁决，归 write_fact 的 confirm 分支。标定流
    （StreamStats）只吃规范值——表面形式的搅动不得计入翻转率。

## 入口

| 入口 | 启动 | 说明 |
|---|---|---|
| 网页 | `python web/server.py [端口] [游移秒]` | 聊天 + 识海 3D 星图 + 记忆面板 + 语音条（模型自主发） |
| 语音终端 | `python main.py` | 开口即语音（CosyVoice，播放中可打断）+ 摄像头视觉 |
| 离线演示 | `python demo_memory_flow.py` | 六个场景把调研结论跑给你看，不调 API |

语音（可选）：CosyVoice3 源码与模型收在 `third_party/`（gitignore），
依赖见 `requirements-voice.txt`；Mac/便携版封装在 `voice/tts_mac.py`
（零硬编码路径，设备自适应 cuda→mps→cpu），Windows 版 `voice/tts.py`
原样保留，`voice/__init__.py` 按平台自动分派。
