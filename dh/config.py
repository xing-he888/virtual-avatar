# -*- coding: utf-8 -*-
"""全部工程参数集中在一个地方，每个数都写明出处。

出处约定：
    [SUMMARY §x]  = PROJECT_SUMMARY.md 第 x 节
    [DESIGN §x]   = DIGITAL_HUMAN_DESIGN.md
    [STORE §x]    = DIGITAL_HUMAN_STORAGE.md
    [GRAPH §x]    = DIGITAL_HUMAN_MEMORY_GRAPH.md
    [AUDIT §x]    = SCORING_FUNCTION_AUDIT.md
    [真实数据]     = 第六阶段 50,465 条微信消息的外部检验

被真实数据推翻的参数不在这里出现（exp(-Δ/S) 全局 S=32 已废除）。
"""
import numpy as np

# ---------------------------------------------------------------- 时间核
# 核形状：对数正态（先升后降）。真实数据 6/6 会话全选对数正态，LR=+2068.9 [SUMMARY §6.3]
#   w(Δ) = exp(-0.5 z²) / Δ,  z = (ln Δ - ln S) / sigma
KERNEL_KIND = "lognormal"
KERNEL_SIGMA = 1.804          # 主会话拟合 [SUMMARY §6.3]
KERNEL_S_DEFAULT = 660.0      # 秒。exp(-0.5 z²) 因子的峰在 0.18 h [SUMMARY §6.3]
                              # 注意：这只是一种渠道（微信私聊）的先验，不是全局常数——
                              # 换渠道必须重标定后冻结 [SUMMARY §6.4，14 会话差 835 倍]

# 段边界项：bnd=1 的行乘 exp(c_bnd * ln(Δ/flip))，<flip 抑制、>flip 促进 [SUMMARY §6.5]
# 符号结构 8/8 会话同号（最稳的一条）；翻转位置只是小时量级（留出 0.29-2.94 h），
# 所以 flip 是参数不是常数，默认取主会话的 0.8 h。
# 原始拟合常数 b_bnd=-0.4913、c_bnd=+0.1900 与「b + c·lnΔ」写法等价于
# flip = exp(-b/c) · (Δ 的单位基准)，这里把位置信息收进 flip、斜率保留 c_bnd。
BND_C = 0.1900                # [SUMMARY §6.5] = c_bnd
BND_FLIP_S = 0.8 * 3600.0     # 0.8 h [SUMMARY §6.5]；留出实测 0.29-2.94 h，只能主张小时量级
SEG_GAP_S = 1800.0            # 段由 >30 min 空隙切开 [SUMMARY §6.5]

# 指数核保留给标定实验（φ/back 回归是在指数核口径下拟合的）
EXP_S_DEFAULT = 32.0          # [SUMMARY 工程落地参数] 冷启动表，仅合成世界口径

# ---------------------------------------------------------------- 标定
# ln S = 4.5345 - 0.2477·ln(phi) + 0.7246·ln(back)   训练 R²=0.8186 [SUMMARY 正面结论 2]
CALIB_REG = (4.5344987989547825, -0.24769243768956906, 0.7246316893732462)
CALIB_N_SMALL = 50            # 每实体观测 <50 条：用默认值 [SUMMARY 启动时间表]
CALIB_N_FULL = 150            # >=150 条：游程直方图（F1）特征 [SUMMARY 启动时间表]
# 之后冻结。在线自适应实测亏（±58% 常数误差只值 2 个百分点）[DESIGN §3.3]

# ---------------------------------------------------------------- 情绪
# OU 动力学参数（lab/emotion/model.py，E2 可辨识性结论约束）
EMO_THETA = np.array([0.10, 0.15, 0.10])   # 返回基线速度（PAD 三维，单位 1/min）
EMO_SIGMA = np.array([0.25, 0.30, 0.20])
EMO_MU = np.zeros(3)                       # 情绪基线 = 性格参数
# 注意：本模型 realised 衰减率比名义 theta 快约 3.4 倍 [SUMMARY 情绪模块]
# 估计衰减率一律用 a = C2/C1（偏差 <=4%），不许拿 C1/C0（rel=0.5 时 +239%）[SUMMARY 正面结论 4]

# 情绪一致检索（情绪路）：score 加 1 + bias·mood·valence 的匹配项 [DESIGN §3.2]
EMO_BIAS = 1.0
# 唤醒度门控只适用于情节/事件记忆，不适用于事实记忆 [DESIGN §4.3 补注，E3/E5/E7]
# 评价事件的注入权重下限：OU 的 (1-a) 会随 tick 粒度缩没，评价按「至少一次
# 速行窗口」的强度进状态（工程选择，保证事件冲击与 tick 粒度无关）
EMO_DRIVE_MIN = 0.15

# ---------------------------------------------------------------- 工作记忆 / 注意
WM_SLOTS = 4                 # 硬上限，第 5 个挤掉显著性最低的 [DESIGN 规则 1]
ATTEND_PER_TICK = 2          # 每 tick 从感觉缓冲进来的容量（硬的，进不来就留在缓冲）
SENSORY_TTL_S = 2.0          # 感觉缓冲约 2 秒 [DESIGN §2]

# ---------------------------------------------------------------- 检索
GATE_K = 8                   # 硬门槛：相关度前 K 条（E41：K=8 留出 0.922 vs 不设 0.069）[AUDIT §6]
TALK_K = 5                   # 每轮讲出来的条数：筐内按分数抽样，不 argmax [GRAPH §9.5]
HYBRID_W = {"sim": 0.5, "time": 0.3, "emo": 0.2}   # 三路权重（名义）；z 标准化后实际占比=名义占比 [AUDIT §2]
SCORE_RTOL = 0.10            # 断言：|实际方差占比 - 名义占比| <= 0.10 [AUDIT §3]

# ---------------------------------------------------------------- 激活图
EDGE_W = {"link": 1.0, "share": 0.7, "sim": 0.5}   # [GRAPH §4]
DIFF_DECAY = 0.55            # 每跳衰减 [GRAPH §3.1]
DIFF_HOPS = 3                # 3 跳最好，5 跳反而降 [GRAPH §3.1]
HUB_DEG_CAP = 20.0           # 出度超过此值的节点边权乘 D/deg（阈值未扫过，诚实默认）[GRAPH §8]
# 实体节点不发散：它的边只用来进入，不参与传递 [GRAPH §4]
ASSOC_ROUNDS = 6
ASSOC_PER_ROUND = 20
ASSOC_BASKET = 200           # 点亮筐 [GRAPH §9.5]
SEEN_RHO = 0.55              # 已想起降权 [GRAPH §9.5]
SIM_EDGE_MIN_COS = 0.45      # 识海显示层的 sim 边阈值 [GRAPH §1]
                             # 0.55 是合成世界标定（那边文本共享词多）；真实
                             # 闲聊同实体事实的哈希余弦 0.37-0.55，0.55 会全军覆没
SPREAD_SIM_MIN_COS = 0.55    # 激活扩散沿 sim 边传播的门槛——画出来的相似网
                             # 不等于点亮会走的路（想起≠蔓延），扩散要收敛：
                             # 老王的事不该把张姨的电话点亮（test_graph 守护）

# ---------------------------------------------------------------- 幻觉治理
# 来源分级 [DESIGN §6 第一层]
REL_LABEL = {0: "亲眼看到", 1: "亲口告诉", 2: "听说", 3: "推断", 4: "先验"}
REL_PHRASE = {
    0: "我看到",
    1: "你告诉过我",
    2: "听{s}说",
    3: "我猜",
    4: "",               # L4 不能当记忆说
}
EXPIRED_PHRASE = "我记得以前是{old}，现在不确定"      # [DESIGN §6 第二层]
ABSTAIN_PHRASE = "这个我记不清了"                     # [DESIGN §6 第三层：放最后做，别期待它]

# ---------------------------------------------------------------- 反应 / 思维
SLOW_BUDGET_TOKENS = 2000    # 慢系统显式预算 [DESIGN §5.1 陷阱提醒]
CONFLICT_THRESH = dict(
    c1_tie=0.15,             # 检索置信度低：最高分与次高分差小于这个比例
    c4_arousal=0.65,         # 情绪强度过高
)
IDLE_WANDER_S = 120.0        # 空闲多久开始心智游移 [DESIGN §5.2]

# ---------------------------------------------------------------- 存储
DB_PAGE_BYTES = 4096


def bnd_multiplier(age_s, seg_bnd, flip_s=None, c=None):
    """段边界乘子：exp(c · ln(Δ/flip) · bnd)。bnd=0 恒为 1。"""
    c = BND_C if c is None else c
    flip_s = BND_FLIP_S if flip_s is None else flip_s
    if not seg_bnd:
        return 1.0
    return float(np.exp(c * np.log(max(age_s, 1e-9) / flip_s)))
