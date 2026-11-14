# -*- coding: utf-8 -*-
"""模拟记忆流程：把调研结论跑给你看。

五个场景对应五份文档的核心结论：
    一、七步循环与五个存储区       -> DIGITAL_HUMAN_DESIGN §1/§2, STORAGE
    二、来源分级与过期降级（幻觉治理）-> DESIGN §6
    三、情绪动力学与情绪一致检索    -> DESIGN §4, MEMORY_GRAPH §10
    四、激活扩散与联想循环          -> MEMORY_GRAPH §3/§9
    五、离线巩固与心智游移          -> DESIGN §5.2 / 期5

运行：
    C:/Users/xinghe/miniconda3/envs/virtual_avatar/python.exe demo_memory_flow.py
全部离线，不调任何 API；同种子同结果。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402

from dh import config, world  # noqa: E402
from dh.core.tick import DigitalHuman  # noqa: E402

SEP = "=" * 68


def head(title):
    print("\n%s\n%s\n%s" % (SEP, title, SEP))


def main():
    rng = np.random.default_rng(7)

    # ================================================================ 一
    head("一、七步循环：它不是问答框，是一直醒着的心智循环")
    dh = DigitalHuman(seed=11)
    print(">>> 张姨说：老王的电话是 13800001234")
    rec = dh.hear("老王的电话是 13800001234", speaker="张姨", t=10.0)
    print("    tick: 注意了 %d 条 | 情绪 P=%+.3f A=%.3f | 写回 1 事实" % (
        rec["attended"], rec["mood"]["p"], rec["mood"]["a"]))
    print(">>> 我说：老王还有个备用号 13900005678 / 13700009999")
    dh.hear("老王还有个备用号 13900005678", speaker="我", t=12.0)
    dh.hear("老王还有个备用号 13700009999", speaker="我", t=13.0)
    print(">>> 我问：老王现在电话是多少")
    ans, phr, diag = dh.ask("老王现在电话是多少", t=20.0)
    print("    小星: %s" % phr)
    print("    （点查走 SQLite 索引，不调模型；冲突信号=%s）" % diag["conflict_flags"])
    print(">>> 我问：老王有什么备用电话")
    ans, phr, _ = dh.ask("老王有什么备用电话", t=21.0)
    print("    小星: %s" % phr)
    print("    <- 谓词自带 multi=1，两个号并存；单值谓词永远只报一个")
    print(">>> 我问：老李的电话是多少（从没听过）")
    ans, phr, _ = dh.ask("老李的电话是多少", t=22.0)
    print("    小星: %s   <- 弃权：检索层没有的，编不出来" % phr)
    print("\n    工作记忆（4 槽硬上限）：")
    for it in dh.wm.items:
        print("      · [%s] salience=%.2f %s" % ("问" if it.get("is_question") else "述",
                                                 it["salience"], it["text"][:24]))

    # ================================================================ 二
    head("二、幻觉治理：来源分级 + 过期检测")
    dh2 = DigitalHuman(seed=12)
    dh2.hear("老王的电话是 13800001234", speaker="张姨", t=10.0)
    dh2.hear("听张姨说，老王老婆住院了", speaker="我", t=20.0)
    dh2.hear("老王换号了，13800005678", speaker="我", t=3000.0)
    dh2.hear("我刚给老王打电话，13800001234", speaker="李医生", t=5000.0)  # 传旧话
    print(">>> 李医生在 5000 分钟又提旧号 1234（世界 2.7% 的轮次在传旧话）")
    ans, phr, _ = dh2.ask("老王现在电话是多少", t=5100.0)
    print("    小星: %s   <- 复述旧值不复活（confirm_old）" % phr)
    ans, phr, _ = dh2.ask("老王老婆病情怎么样了", t=100.0)
    print(">>> 「听张姨说，老王老婆住院了」之后再问:")
    print("    小星: %s" % phr)
    print("    <- 来源留最早那条（张姨），不是最近说话人 [e50f §六：取最早=1.000]")
    ans, phr, _ = dh2.ask("老王现在电话是多少", t=2500.0)
    print(">>> 回到 2500 分钟问（时间旅行）:")
    print("    小星: %s   <- 区间查询 valid_from/valid_to，不信 status" % phr)

    # ================================================================ 三
    head("三、情绪：PAD 动力学 + 情绪一致检索")
    dh3 = DigitalHuman(seed=13)
    print(">>> 住院期（效价 -0.75 的轮次连续输入）：")
    t = 20000.0
    for i in range(8):
        dh3.hear("老王老婆还在住院，第%d天" % (i + 1), speaker="张姨", t=t + i * 60,
                 valence=-0.75)
    m = dh3.affect.state
    print("    情绪被拖低：P=%+.3f A=%.3f D=%+.3f" % (m.p, m.a, m.d))
    print(">>> 中间也有好事（出院，效价 +0.6）：")
    for i in range(4):
        dh3.hear("老王老婆出院了，恢复得不错", speaker="我", t=t + 600 + i * 60,
                 valence=0.6)
    print(">>> 之后不再有新事件（每 30 分钟看一眼）：")
    line = []
    for k in range(1, 5):
        for _ in range(30):
            dh3.affect.tick(60000.0, rng=rng)      # 每次一分钟
        line.append("%3dmin P=%+.2f" % (30 * k, dh3.affect.state.p))
    print("    " + "  |  ".join(line))
    print("    <- OU 动力学回基线；不精确停在 0 是 σ 项（人的情绪不会归零锁死）")
    print("    四出口各读各的维：语言 warmth=%+.2f | 语音 rate=%.2f | 表情 au4=%.2f" % (
        dh3.affect.outlets()["language"]["warmth"],
        dh3.affect.outlets()["speech"]["rate"],
        dh3.affect.outlets()["face"]["au4"]))
    print(">>> 情绪一致检索（同一份记忆，两种心情；情绪路在情节层召回）：")
    picks = {}
    for mood in (-1.0, 1.0):
        res = dh3.retriever.recall(query_text="老王老婆", mood=mood, t_now=t + 900)
        vals = []
        for kind, rid, _ in res[:10]:
            if kind == "obs":
                vals.append(float(dh3.store.obs_row(rid)["valence"]))
            else:
                vals.append(dh3.retriever._fact_valence(rid))
        picks[mood] = float(np.mean(vals))
        print("    心情%+d：想起 10 条的平均效价 %+.3f" % (int(mood), picks[mood]))
    spread = picks[1.0] - picks[-1.0]
    print("    差 %+.3f%s <- 不用写规则，「心情不好想起糟心事」自己浮现" % (
        spread, "（>0）" if spread > 0 else "（<=0，本场景效价样本不足）"))

    # ================================================================ 四
    head("四、激活扩散与联想循环（ρ=0.55 / 筐 K=200 / 按分数抽样）")
    # E48 的合成世界：1200 条记忆、40 个人、目标人物只有 8 条、3 个枢纽各 180 条。
    # （30 天世界的事实行按值变动去重后只有 22 行，是点查的功劳，
    #   但对联想演示来说节点太少——联想演示用 E48 同款记忆库。）
    from dh.memory.store import MemoryStore
    from dh.memory.graph_layer import ActivationGraph
    s4 = MemoryStore()
    s4.declare_pred("电话", 0)
    NF, NP, TARGET = 1200, 40, 7
    _rng = np.random.default_rng(101)
    ppl = _rng.integers(0, NP, NF)
    hub = _rng.choice([p for p in range(NP) if p != TARGET], 3, replace=False)
    ppl[:180] = _rng.choice(hub, 180)
    ppl[ppl == TARGET] = (TARGET + 1) % NP
    own = _rng.choice(NF, 8, replace=False)
    ppl[own] = TARGET
    _rngv = np.random.default_rng(5)
    for i in range(NF):
        o = s4.add_obs(float(i), "told", "text", "记忆%d" % i, source="张姨")
        fid, _ = s4.write_fact("P%02d" % ppl[i], "电话", "138%08d" % i, float(i), o)
        v = _rngv.standard_normal(64).astype(np.float32)
        v /= np.linalg.norm(v)
        # model 记账：这 64 维是随机向量、不是任何嵌入模型的输出，
        # 单独标一个名字，免得跟 bge 的 512 维混在一次 load_matrix 里
        s4.put_vec("fact", fid, v, model="demo-random64")
    g4 = ActivationGraph(s4)
    # own 是数组下标，fact_id = 下标 + 1。写错这一行，种子就指向别人——
    # 之前打印的 58/0.517 就是错种子 + 零权筐两个 bug 叠出来的
    seeds = {int(own[0]) + 1: 1.0, int(own[1]) + 1: 1.0}
    res = g4.associate(seeds, np.random.default_rng(4), rounds=6, per_round=20,
                       basket=config.ASSOC_BASKET)
    print("    种子：目标人物的 2 条记忆 -> 连想 6 轮 × 20 条")
    print("    累计触达 %d 条不同记忆，重复率 %.3f   [E48 文档：87.0 / 0.275]" % (
        res["touched"], res["repeat_rate"]))
    r_no = g4.associate(seeds, np.random.default_rng(4), rounds=6, per_round=20,
                        basket=config.ASSOC_BASKET, rho=1.0)
    print("    对照 ρ=1.0（不降权）：触达 %d，重复率 %.3f   <- 已想起降权是必需的" % (
        r_no["touched"], r_no["repeat_rate"]))
    a20 = g4.associate(seeds, np.random.default_rng(1), rounds=6, per_round=20,
                       basket=20, rho=0.55, how="argmax")
    a200 = g4.associate(seeds, np.random.default_rng(1), rounds=6, per_round=20,
                        basket=200, rho=0.55, how="argmax")
    print("    对照 argmax：筐 20 与筐 200 的选取逐位相同（%s），触达 %d"
          " <- argmax 废掉筐" % (a20["picks"] == a200["picks"], a200["touched"]))

    # ================================================================ 五
    head("五、离线巩固 + 心智游移")
    turns = world.build_world(20261031)
    dh4 = DigitalHuman(seed=14)
    for tr in turns[:600]:
        dh4.hear(tr["text"], speaker=tr["speaker"], t=float(tr["t"]),
                 valence=tr["valence"])
    dh4.store.commit()
    res = dh4.idle(t=601.0, rng=np.random.default_rng(4))
    print("    心智游移（空闲触发，产出 rel=3 只能说「让我想起」）：")
    for th in res["wander"]["thoughts"][:4]:
        print("      · %s" % th[:40])
    print("    触达 %d 条 | 重复率 %.3f | 写回 %d 条 infer 观测" % (
        res["wander"]["touched"], res["wander"]["repeat_rate"],
        dh4.store.con.execute("SELECT COUNT(*) FROM obs WHERE kind='infer'").fetchone()[0]))
    print("    自我叙事（极慢，只在强烈重复事件上更新）：")
    for row in dh4.store.self_narrs():
        print("      · %s" % row[1])

    # ================================================================ 六
    head("六、30 天对话世界端到端（tests/test_world_e2e.py 的完整协议）")
    print("    1800 轮灌入七步循环，105 个留出查询按金标记分（+1/-3/-1/0）。")
    print("    文档 P0（来源前缀不剥+argmax）：留出效用 +0.6827   <- 本实现复刻值相同")
    print("    文档 P2（剥来源+多值声明）  ：留出效用 +0.9519 / 完全答对 0.981")
    print("    本实现（双时态存储+点查）  ：留出效用 +1.0000 / 完全答对 1.000")
    print("    <- 存储层的 supersede 语义比核加权投票干净；「说过期」压到 0")

    print("\n%s\n完成。所有参数与出处见 dh/config.py。\n%s" % (SEP, SEP))


if __name__ == "__main__":
    main()
