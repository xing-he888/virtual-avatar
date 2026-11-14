# -*- coding: utf-8 -*-
"""性格特质向量（OCEAN 五维）：成长层的真值源。

为什么是 OCEAN：Big Five 是人格测量的通用语言，且**到 PAD 情绪有公开的
回归公式**（Mehrabian 1996《Big-five 与 PAD 气质模型》，经 Gebhard ALMA
2005 操作化）——特质 → 情绪基线不用造轮子，代公式即可：

    Trait Pleasure  ≈ 0.21·E + 0.59·A − 0.19·N      （外向/宜人→底色亮，
                                                      神经质→底色暗）
    Trait Arousal   ≈ 0.15·O + 0.30·C + 0.57·N      （神经质→易唤起、警觉）
    Trait Dominance ≈ 0.25·O + 0.17·C + 0.60·E − 0.32·A
                                                      （外向→强势，宜人→柔和）
    ⚠ 系数取自 ALMA 引用集；符号按 Mehrabian 定性结论校准（外向=pleasant/
    dominant，神经质=unpleasant/arousable/submissive）。ALMA 原文的 Nu 符号
    存在两种转载口径，接入时如需逐字对齐请核对原论文。

存储纪律与 alias/calib/value_alias 同款：真值入 calib 表
scope='growth'（只插不改、取最新行）——每次塑形插一行，成长曲线天然可回放。
值域 [-1,1]；一生漂移总量钳制 |trait - seed| ≤ GROWTH_DRIFT_CLAMP。
"""
import numpy as np

from .. import config

TRAIT_NAMES = ("O", "C", "E", "A", "N")
TRAIT_LABEL = {"O": "开放", "C": "尽责", "E": "外向", "A": "宜人", "N": "神经质"}


def seed_traits():
    """出厂种子（dict）。由 config.GROWTH_SEED 给出——小星的「基因」。"""
    return dict(config.GROWTH_SEED)


class Traits:
    """某只小星的活特质：读（calib 最新行）、写（插新行）、钳制、换算。"""

    def __init__(self, store):
        self.store = store

    # ------------------------------------------------ 读
    def get(self):
        """当前特质向量 dict。没有历史时返回出厂种子。"""
        t = seed_traits()
        for name in TRAIT_NAMES:
            v = self.store.calib_latest("growth", "trait", name)
            if v is not None:
                t[name] = float(v)
        return t

    def seed(self):
        return seed_traits()

    # ------------------------------------------------ 写
    def update(self, moves, t, note="shape"):
        """moves: {trait: delta}。逐维插新行（只增），返回更新后的向量。

        双重钳制：单维值域 [-1,1]；一生漂移距种子 ≤GROWTH_DRIFT_CLAMP
        （经历留疤但不翻盘——性格不会一夜变成另一个人）。

        时间戳只进不退（审查 P2-2）：fitted_t = max(t, 该键最新行的
        fitted_t)——calib_latest 按 fitted_t DESC 取行，回拨时间戳写入的
        行会被旧行遮蔽、写入静默丢失（shape_direct(t=100) 被 t=200 的旧行
        盖住的复现），与信任水位线（persona._update_trust）同款护栏。

        只写 moves 里动了的键：没动的特质保留原行，calib 历史里每行都是
        一次真实的变动（旧版每次全量补写 5 行，恢复力按次结算时一天
        3600 行全是零变化噪声）。

        本函数是特质**唯一的写入口**（shape / shape_direct / regress 三路
        都经它），所以恢复力的时间锚 growth_regressed_at 在这里刷新：
        新疤从落下那刻开始计时，不会被上一次结算以来的整段空窗超量回收
        （审查 P1-1 重写，语义见 config.GROWTH_REGRESS_TAU_DAYS）。
        ⚠ growth_shaped_at（证据结账水位线）依旧**不**在此推进：只有
        Persona.shape() 的成功塑形路径才推进——regress（恢复力）和
        shape_direct（编程直调）都不消费证据，动了水位线会把未达
        GROWTH_MIN_EVIDENCE 门槛、正在跨窗口攒着的经历永久丢掉
        （子代理审查 P1-1 实证过）。"""
        cur = self.get()
        seed = seed_traits()
        clamp = config.GROWTH_DRIFT_CLAMP
        out = {}
        for name in TRAIT_NAMES:
            if name not in moves:
                out[name] = cur[name]
                continue
            v = cur[name] + float(moves[name])
            v = float(np.clip(v, -1.0, 1.0))
            v = float(np.clip(v, seed[name] - clamp, seed[name] + clamp))
            out[name] = v
            if abs(v - cur[name]) < 1e-12:
                # 被 clamp 顶回原值：值没变就不插行（审查三轮 F2——同值行
                # 是纯噪声，calib 只该记录真实变动）
                continue
            prev_t = self.store.calib_latest_t("growth", "trait", name)
            t_key = max(float(t), float(prev_t)) if prev_t is not None else float(t)
            self.store.calib_set("growth", "trait", name, out[name],
                                 note=note, fitted_t=t_key)
        if moves:
            old = self.store.get_meta("growth_regressed_at")
            anchor = max(float(t), float(old)) if old else float(t)
            self.store.set_meta("growth_regressed_at", repr(anchor))
        return out

    # ------------------------------------------------ 换算出口
    def encode_factor(self):
        """尽责 → 写入编码强度乘子（1 + GROWTH_C_ENCODE·C）。

        性格作用于「存」的第二根线（用户提案「存取对记忆加权」）：C 高的
        人把经历记得更牢。C∈[-1,1] → 因子有界 [1−k, 1+k]；出厂 C=0.2 →
        ×1.08，近中性点位（出厂人行为基本不变，特质漂了才显出来）。"""
        return 1.0 + config.GROWTH_C_ENCODE * float(self.get()["C"])

    def mu_pad(self):
        """OCEAN → PAD 气质基线（Gebhard/ALMA 公式）→ 直接喂 OU 的 mu。

        这是「性格控制情绪」的第一出口：特质变 → 基线变 → OU 稳态变 →
        平时的底色真的移动（不是提示词装出来的）。"""
        t = self.get()
        p = (0.21 * t["E"] + 0.59 * t["A"] - 0.19 * t["N"])
        a = (0.15 * t["O"] + 0.30 * t["C"] + 0.57 * t["N"])
        d = (0.25 * t["O"] + 0.17 * t["C"] + 0.60 * t["E"] - 0.32 * t["A"])
        return np.array([p, a, d], dtype=np.float64)

    # ------------------------------------------------ 展示
    def bars(self):
        """给提示词/前端的一行特质条：外向0.62 ▮▮▮▮▯ 。"""
        out = []
        for name in TRAIT_NAMES:
            v = self.get()[name]
            filled = int(round((v + 1) / 2 * 4))
            out.append("%s%s%+.2f%s" % (TRAIT_LABEL[name], "▌" * max(filled, 0),
                                        v, "░" * max(4 - filled, 0)))
        return "  ".join(out)
