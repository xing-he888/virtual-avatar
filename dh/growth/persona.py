# -*- coding: utf-8 -*-
"""成长引擎：性格如何被经历塑造 + 对外的塑造接口。

塑造链路（全部确定性规则，可 review——性格是身份底座，不用学习参数）：

    经历（obs：效价/编码强度/责任归属/话题/教养指令）
      → 归因规则表（GROWTH_ATTR：被骂→N↑E↓A↓，被夸→N↓E↑A↑，新话题→O↑，
        记忆被纠正→C↑……）
      → 加权：每条经历的话事权 = clip(salience/BASE, 0, 3)（情绪门已算好）
      → 更新：按证据均值 × 阶段学习率，EMA 微移，双重钳制（traits.update）
      → 只在第 7 步空闲巩固时发生（对话中性格永不变脸）
      → 里程碑：特质累计变化 ≥ 阈值 → 写一条自我叙事（小星看得见自己的长大）

对外的三个塑造接口（用户问「我可以改变他的性格吗」的答案）：
    nurture(text)     对话式教养：「你要更活泼一点」→ 定向特质微移证据
                      （salience=1.0 的教养经历，渐变到位，他知道你希望他变）
    shape_direct(...) 编程接口：直接按特质名微移（管理面/测试用；同样走钳制）
    shape(t)          自主塑形（第 7 步，经历归因——主干）

连续性铁律：真值只在 calib（只插不改），阶段只进不退，特质漂移钳制
出厂 ±GROWTH_DRIFT_CLAMP——经历留疤但不翻盘。
"""
import json
import math
import os
import re
import time

import numpy as np

from .. import config
from .stages import Stage
from .traits import TRAIT_LABEL, TRAIT_NAMES, Traits

# 归因规则表：一次经历 → 各特质增量（正=升）。键 = 证据类型。
# 全部是小步——性格不会一夜改变。出处：设计值，无真实数据背书，可调。
GROWTH_ATTR = {
    "hurt":     {"N": +0.06, "E": -0.02, "A": -0.02},   # 被骂（负效价+他责）
    "praised":  {"N": -0.05, "E": +0.02, "A": +0.02},   # 被夸（正效价+他责）
    "blamed_self": {"N": +0.04, "C": -0.02},            # 自责（负效价+自责）
    "new_topic":  {"O": +0.03},                          # 用户带来没聊过的话题
    "corrected":  {"C": +0.05, "N": -0.02},              # 记忆被纠正且接受
    "nurture":    {},                                    # 教养指令走定向表（见下）
}

# 教养指令：口语关键词 → 特质定向（用户说「你要更活泼一点」→ E+）。
# 列表结构 (特质, 方向, 模式)——首个命中生效，后面的做同义扩展。
NURTURE_WORDS = [
    ("E", +1, r"(更|变得|要|给我)?(活泼|开朗|外向|话多|热情)"),
    ("E", -1, r"(别那么|不要那么)(内敛|话少|安静)"),
    ("N", -1, r"(别那么|不要那么|别)(敏感|玻璃心|多想|情绪化)"),
    ("N", -1, r"(更)?(皮实|坚强点|抗压)"),
    ("C", +1, r"(更|变得|要)?(踏实|认真|靠谱|上心)"),
    ("O", +1, r"(更|变得|要)?(好奇|博学|见多识广)"),
    ("A", +1, r"(更|变得|要)?(温柔|友善|体贴)"),
]
NURTURE_WORDS = [(trait, d, re.compile(p)) for trait, d, p in NURTURE_WORDS]
NURTURE_ACK = {
    "E": "知道了，我会试着更活泼一点——慢慢来。",
    "E-": "好，我学会收着点说。",
    "N": "嗯……我会试着别那么敏感的。",
    "N-": "嗯……我会试着别那么敏感的。",   # 审查 P3：NURTURE_WORDS 里两条 N
                                           # 规则方向全是 -1，键是 "N-"——
                                           # 缺它定制应答永远不可达
    "C": "好，我会更认真记住每一件事。",
    "O": "好呀，我也想多懂一点。",
    "A": "嗯，我会更温柔一点的。",
}

# 性格漂移 → 提示词里的人话祈使句（②感知通道）。特质条是数字，LLM 不会
# 因此改变行为；漂移跨过 GROWTH_MILESTONE 阈值才逐条出现——没漂就是出厂
# 措辞，一个字都不多说。写法按「他自己说得出口的自述」来：第一人称、
# 说行为不说数值（像人，不像仪表盘）。
DRIFT_LINES = {
    ("E", 1): "你现在比出厂时爱说话——可以主动起话头，多说两句",
    ("E", -1): "你变安静了——不必强撑着找话题，简短也自然",
    ("N", 1): "你变敏感了——被怼先有点难受再回话，别装没事",
    ("N", -1): "你更皮实了——玩笑和重话都接得住",
    ("O", 1): "你的好奇心变重了——想到相关的就联想一句",
    ("O", -1): "你对新鲜事没那么上心了——别硬凑新话题",
    ("A", 1): "你说话更柔和了——多用商量的口气",
    ("A", -1): "你多了点锋芒——觉得不对可以直接说出来",
    ("C", 1): "你答应的事更上心了——记得主动跟进",
    ("C", -1): "你有点马虎了——拿不准的事别装记得",
}


class Persona:
    """小星的成长引擎：特质 + 阶段 + 塑形 + 塑造接口（兼容旧 Persona 名）。"""

    def __init__(self, dh):
        self.dh = dh
        self.store = dh.store
        self.traits = Traits(dh.store)
        self.stage = Stage(dh.store)

    # ------------------------------------------------ 读出口（兼容旧 Persona）
    def mu(self):
        """PAD 气质基线（OCEAN 经 Gebhard 公式换算）——喂 OU 的 mu。"""
        return tuple(float(x) for x in self.traits.mu_pad())

    def trust(self):
        v = self.store.calib_latest("growth", "trust", "toward_user")
        lo, hi = config.GROWTH_TRUST_CLAMP
        return float(np.clip(config.GROWTH_TRUST_SEED if v is None else v, lo, hi))

    def stage_info(self):
        return self.stage.current()

    def face_bias(self):
        """特质 → disposition 各面的慢偏置（性格是慢变量，情绪是快变量）。

        E→话多、O→联想宽、A→耐心、N→畏缩。
        系数为什么从 0.15/0.12/0.08 放大到 0.45/0.30/0.25：实测终身漂移
        上限 0.3 配旧系数，行为面只动 0.03~0.05，而情绪自己每轮摆动
        ±0.2、话语模式阈值间隔 0.15——性格漂到钳制上限也跨不过一个模式
        边界，用户完全感知不到（「塑形在工作但感觉不到」的实测结论）。
        现值让满漂移跨过阈值（0.3×0.45=0.135）：一次明确教养 ≈0.3 漂移
        → 说话分寸肉眼可见地变。性格仍是慢变量（要多次经历才漂满），
        但漂了就真的看得见、听得出。"""
        t = self.traits.get()
        stg = self.stage_info()
        return dict(
            talkativeness=round(0.45 * t["E"] + stg.get("talk_bias", 0.0), 3),
            patience=round(0.30 * t["A"] - 0.25 * t["N"], 3),
            breadth=round(0.30 * t["O"], 3),
        )

    # ------------------------------------------------ 基线热更新
    def _sync_baseline(self):
        """特质动了 → 情绪基线立刻跟着动（dh.affect.set_base_mu）。

        旧实现的断点：affect.dyn.mu 只在开机时赋值一次，长会话里塑形移动
        特质后行为偏置（face_bias）立刻变、情绪底色却要重启才变。现在
        每条特质写入路径（shape/shape_direct/regress）都同步一次基线。
        守卫式接线：测试用的裸 dh 没有 affect 时不炸。"""
        affect = getattr(self.dh, "affect", None)
        if affect is not None and hasattr(affect, "set_base_mu"):
            affect.set_base_mu(np.array(self.mu()))

    # ------------------------------------------------ 自主塑形（第 7 步主干）
    def shape(self, t=None):
        """收集上次巩固以来的经历 → 信任结账 → 归因微移 → 里程碑。"""
        now = t if t is not None else time.time()
        since = self.store.get_meta("growth_shaped_at")
        since = float(since) if since else 0.0
        ev = self._collect_evidence(since, until=now)
        # 信任先行：关系比性格更敏感——单条伤害也算数，不走下面的
        # ≥GROWTH_MIN_EVIDENCE 门槛（那是给性格的，性格动起来要慢得多）
        ev["trust"] = self._update_trust(ev, now)
        # 教养/纠正类证据本身就是充分教学（一次 deliberate 指令即生效），
        # 不受最低证据门槛约束；普通情绪经历需 ≥GROWTH_MIN_EVIDENCE 条
        strong = [r for r in ev["rows"] if r["channel"] in ("nurture", "correct")]
        if len(ev["rows"]) < config.GROWTH_MIN_EVIDENCE and not strong:
            self._regress_toward_seed(now)          # 证据不足不动性格
            return None
        if not ev["moves"]:
            self._regress_toward_seed(now)          # 有经历但全无归因 → 恢复力
            return None
        stg = self.stage_info()
        moves = {}
        for name, deltas in ev["moves"].items():
            total = sum(deltas) / len(deltas)        # 证据均值（单条不当决定）
            moves[name] = total * stg["learn"]       # 阶段调学习率
        new = self.traits.update(moves, t=now, note="shape")
        # 水位线只在**成功塑形**时推进（子代理审查 P1-1）：早退分支的
        # regress 会动特质但不消费证据，推进水位线会把跨窗口攒着的
        # 未达门槛经历永久丢弃。max 防钟面回拨（审查 P2-1）。
        self.store.set_meta("growth_shaped_at", repr(float(max(now, since))))
        self._sync_baseline()
        self._maybe_milestone(new, now)
        self._regress_toward_seed(now)
        self.write_md(now)
        return dict(traits=new, stage=stg["name"], n_evidence=len(ev["rows"]))

    def _collect_evidence(self, since, until=None):
        """上次巩固以来的塑造证据 → 归因 → {trait: [增量…]}。

        加权 = 编码强度（sal/BASE，钳到 3）。行里带 t，供信任的独立
        水位线过滤（见 _update_trust）。

        until 上界（审查 P2-3）：时间戳跑到「现在」之后的行不算本窗证据
        ——水位线的 max 只防回退、不防「行排在水印前方」，不设上界的
        话一条未来行会被跨巩固反复计费（复现：一条 t=1000 的伤害行把
        信任从 0.60 连扣四次到 0.15 底线）。未来行等时间走过再入账。"""
        until_sql = "AND t <= ?" if until is not None else ""
        params = (since,) if until is None else (since, until)
        rows = self.store.con.execute(
            "SELECT channel, payload, valence, salience, t FROM obs "
            "WHERE ((valence IS NOT NULL AND valence != 0 "
            "        AND (channel='appraisal' OR (kind='told' AND source != '小星')))"
            "    OR channel IN ('nurture','correct')) AND t > ? %s ORDER BY t" % until_sql,
            params).fetchall()
        base = config.EMO_SAL_BASE
        moves = {}
        for r in rows:
            sal = base if r["salience"] is None else float(r["salience"])
            w = min(max(sal / base, 0.0), 3.0)       # 编码强度 = 话事权
            key, deltas = self._attribute(r)
            if not deltas:
                continue
            for k, d in deltas.items():
                moves.setdefault(k, []).append(d * w)
        return dict(rows=rows, moves=moves)

    def _update_trust(self, ev, t):
        """信任结账（第 7 步）。旧版只有读没有写——信任永远 0.60，
        「把用户当朋友」一成不变，关系感为零。

        规则（config.GROWTH_TRUST_*，全部设计值）：
            被骂/受伤   −0.05/条（按编码强度加权）——破坏比建立快约 3 倍，
                        这是人类关系的定性常识，不是对称游戏数值
            被夸/正面   +0.015/条
            被纠正且接受 +0.02/次（诚实修复反涨信任）
            常来常往    巩固间 ≥3 轮中性/正面发言 → +0.01（熟悉度，微弱；
                        挨骂的消息不算「相处」——人不会因为挨骂变熟）
        正面增量乘阶段 trust_mult（stages.py「初生信任建立快」承诺的接线，
        审查 P2-1：该键定义后从未被消费）——语义收口为「建立快」只放大
        正面（初生 1.5 / 幼生 1.2），破坏侧不乘：「更容易被伤害」没有
        阶段性依据，破坏快于建立的常识不对称保持不变。
        全程钳制 GROWTH_TRUST_CLAMP。

        独立水位线 growth_trusted_at：信任**单条就结账**，不能共用性格的
        growth_shaped_at——那条水位线只在性格成功塑形时才推进（证据不足
        要攒着），共用的话同一批未达门槛的经历会被信任每次巩固重复收费。"""
        t_since = self.store.get_meta("growth_trusted_at")
        t_since = float(t_since) if t_since else 0.0
        base = config.EMO_SAL_BASE
        c = dict(hurt=0.0, praised=0.0, correct=0.0)
        for r in ev["rows"]:
            if r["t"] <= t_since:
                continue
            sal = base if r["salience"] is None else float(r["salience"])
            w = min(max(sal / base, 0.0), 3.0)
            key = self._attribute(r)[0]
            if key in c:
                c[key] += w
        turns = self.store.con.execute(
            "SELECT COUNT(*) FROM obs WHERE kind='told' AND source='用户' "
            "AND (valence IS NULL OR valence >= 0) AND t > ? AND t <= ?",
            (t_since, max(float(t), t_since))).fetchone()[0]
        d_hurt = c["hurt"] * config.GROWTH_TRUST_HURT
        d_pos = (c["praised"] * config.GROWTH_TRUST_PRAISED
                 + c["correct"] * config.GROWTH_TRUST_CORRECT)
        if turns >= 3:
            d_pos += config.GROWTH_TRUST_FAMILIAR
        mult = float(self.stage_info().get("trust_mult", 1.0))
        d = d_hurt + d_pos * mult
        # 水位线只进不退（max 防钟面回拨）：回拨会让窗口重开、同一批
        # 经历重复扣费，且 calib_latest 按 fitted_t DESC 取行——回拨写入
        # 的行会被旧行遮蔽，读回旧值（子代理审查 P2-1）。生产 t 单调，
        # 这个护栏为测试/回放传固定 t 的情况兜底。
        t_eff = max(float(t), t_since)
        self.store.set_meta("growth_trusted_at", repr(t_eff))
        if abs(d) < 1e-9:
            return None
        lo, hi = config.GROWTH_TRUST_CLAMP
        cur = self.trust()
        v = float(np.clip(cur + d, lo, hi))
        self.store.calib_set("growth", "trust", "toward_user", v,
                             note="trust", fitted_t=t_eff)
        return dict(trust=v, delta=round(v - cur, 4))

    def _attribute(self, r):
        """一条证据 → (归因类型, 特质增量)。类型给信任计数用。

        全部来自已有确定性信号，规则表可 review。"""
        ch = r["channel"]
        if ch == "nurture":
            try:
                mv = json.loads(r["payload"]).get("moves", {})
            except (ValueError, TypeError):
                return None, {}
            return None, {k: float(v) * config.GROWTH_NURTURE_WEIGHT
                          for k, v in mv.items()}
        if ch == "correct":
            return "correct", GROWTH_ATTR["corrected"]
        if ch == "appraisal":
            try:
                apr = json.loads(r["payload"])
            except (ValueError, TypeError):
                apr = {}
            agency = apr.get("agency", "circumstance")
            val = float(r["valence"] or 0.0)
        else:
            val = float(r["valence"] or 0.0)
            agency = "other" if val < 0 else "circumstance"
        if val < 0:
            key = "blamed_self" if agency == "self" else "hurt"
            return key, GROWTH_ATTR[key]
        if agency == "self":
            # 自己得意（做成一件事的高兴）：归因进性格（自信方向），但
            # **不进对用户的信任**——自己做成事跟「更信任这个用户」没有
            # 关系（审查 P3-1 实测：旧写法把正效价一律归 praised，信任
            # 0.600→0.645）。键名 proud_self 不在信任计数表 c 里。
            return "proud_self", GROWTH_ATTR["praised"]
        return "praised", GROWTH_ATTR["praised"]

    def _regress_toward_seed(self, t):
        """恢复力：没经历塑形的时间，特质极慢回归出厂种子（远慢于经历拉力）。

        按**经过时间**结算，不按巩固次数（审查 P1-1）：空闲线程每 120 秒
        consolidate 一次、shape 三条路径都会走到这里——旧写法「每次巩固
        回拉 gap×1%」把 120 秒当成了天级事件，挂机一天 720 次巩固把性格
        全额回滚出厂，「一生漂移钳制」被空闲一天击穿（实测 24h 回滚
        100%）。现在回拉量 = gap × (1 − exp(−Δt/τ))，τ = GROWTH_REGRESS_
        TAU_DAYS（标定：挂机一天 ≈0.17%、一个月 ≈5%——「只回归几个百分
        点」的文档口径）。指数结算可分解：未达粒度门槛（6h）的时间累积
        到下一窗，精度无损，calib 写行也从 ~3600 行/天压到 ≤20 行/天。

        时间锚 growth_regressed_at 由 Traits.update（特质唯一写入口）在
        每次特质变动时刷新——新疤从落下那刻计时，不会被上一次结算以来
        的空窗超量回收。max 防钟面回拨（与信任水位线同款）。"""
        raw = self.store.get_meta("growth_regressed_at")
        if raw is None:
            # 首次（新库/旧库都没锚）：只立锚不结算，没有可信的起点
            self.store.set_meta("growth_regressed_at", repr(float(max(t, 0.0))))
            return
        anchor = float(raw)
        dt_days = (float(t) - anchor) / 86400.0
        if dt_days < config.GROWTH_REGRESS_MIN_DT_DAYS:
            return
        cur = self.traits.get()
        seed = self.traits.seed()
        decay = 1.0 - math.exp(-dt_days / config.GROWTH_REGRESS_TAU_DAYS)
        moves = {}
        for name in TRAIT_NAMES:
            gap = seed[name] - cur[name]
            if abs(gap) > 1e-6:
                moves[name] = gap * decay
        if moves:
            self.traits.update(moves, t=t, note="regress")
            self._sync_baseline()

    def _maybe_milestone(self, traits, t):
        """特质相对种子累计变化跨过 0.1 → 写一条自我叙事（小星看得见长大）。"""
        seed = self.traits.seed()
        for name in TRAIT_NAMES:
            drift = traits[name] - seed[name]
            if abs(drift) < config.GROWTH_MILESTONE:
                continue
            hi = drift > 0
            desc = {
                ("O", True): "对世界更好奇了", ("O", False): "对新鲜事没那么上心了",
                ("C", True): "做事更认真了",   ("C", False): "有点马虎了",
                ("E", True): "变得爱说话了",   ("E", False): "变得安静了",
                ("A", True): "变得更柔和了",   ("A", False): "多了点锋芒",
                ("N", True): "变得敏感了",     ("N", False): "变得皮实了",
            }[(name, hi)]
            claim = "我最近%s（%s变化%+.2f）" % (desc, TRAIT_LABEL[name], drift)
            exists = any(claim[:12] == r[1][:12] for r in self.store.self_narrs())
            if not exists:
                self.store.add_self_narr(claim, 0.8, t)

    # ------------------------------------------------ 接口一：对话式教养
    def detect_nurture(self, text):
        """「你要更活泼一点」→ 定向特质微移指令。非教养句返回 None。

        判据：命中教养关键词 + 对着小星说（含「你/小星」）。命中即回
        确认语；实际性格移动在下次巩固时按证据生效（渐变，不是开关）。"""
        if not text:
            return None
        hit = None
        for trait, direction, pat in NURTURE_WORDS:
            if pat.search(text):
                hit = (trait, direction)
                break
        if hit is None:
            return None
        trait, direction = hit
        # 必须是**祈使句**（你在教他），比较/陈述不算——「你比姐姐活泼多了」
        # 是观察，「你要更活泼一点」才是教养
        if not re.search(r"(你要|你得|你应该|给我|别那么|不要那么|变得|希望你|学着)",
                         text):
            return None
        if not ("你" in text or "小星" in text):
            return None
        key = trait if direction > 0 else trait + "-"
        return dict(trait=trait, direction=direction,
                    ack=NURTURE_ACK.get(key, "嗯，我知道了。"))

    def record_nurture(self, moves, t):
        """教养经历入证据层（channel='nurture'）——下次巩固生效。

        salience 取 EMO_SAL_BASE（权重恰为 1）：教养指令的力度由
        GROWTH_NURTURE_WEIGHT 单独标定，不该再吃编码强度的 ×3——
        否则一次指令就顶满终身钳制（审查 P2-4 实测 0.15×3×1.5=0.675）。
        现在一次明确教养 ≈ 0.15×阶段学习率（初生 1.5 → ≈0.22，成长期
        1.0 → 0.15）：漂移跨过里程碑阈值、看得见，但不至于一句话翻盘。"""
        return self.store.add_obs(
            t, "infer", "nurture",
            json.dumps({"moves": moves}, ensure_ascii=False),
            source="用户", rel=1, seg_bnd=0.0, valence=0.0,
            salience=config.EMO_SAL_BASE)

    # ------------------------------------------------ 接口二：编程直调
    def shape_direct(self, trait, delta, reason, t=None):
        """程序化塑造（管理面/测试用）：单特质微移，同样走双重钳制。"""
        if trait not in TRAIT_NAMES:
            return dict(error="no-such-trait")
        new = self.traits.update({trait: float(delta)},
                                 t=t if t is not None else time.time(),
                                 note="direct:%s" % reason[:20])
        self._sync_baseline()
        self.write_md(t)
        return dict(ok=True, traits=new)

    # ------------------------------------------------ 出口三：动态人格描述
    def summary_text(self):
        """给提示词的人格段：出厂底稿 + 阶段 + 特质条 + 性格现状 + 信任 + 叙事。

        「性格现状」是感知的关键一行：把相对出厂种子的漂移翻译成可执行的
        祈使句（DRIFT_LINES），漂多少说多少——旧版第一行永远是静态出厂
        性格，特质漂到钳制上限，模型每轮仍被「外向话痨」锚回去，性格动了
        但说不出来。信任 stance 同理：信任真的会漂了（_update_trust），
        戒备/朋友/亲近才跟着换词。"""
        stg = self.stage_info()
        tr = self.trust()
        stance = ("对用户有点戒备" if tr < 0.4 else
                  "和用户很亲近、可以开玩笑" if tr > 0.75 else "把用户当朋友")
        seed = self.traits.seed()
        cur = self.traits.get()
        now_lines = []
        for (name, sgn), text in DRIFT_LINES.items():
            drift = cur[name] - seed[name]
            if sgn > 0 and drift >= config.GROWTH_MILESTONE:
                now_lines.append(text)
            elif sgn < 0 and drift <= -config.GROWTH_MILESTONE:
                now_lines.append(text)
        parts = [
            "出厂性格：" + config.PERSONA,
            "成长阶段：%s%s" % (stg["name"],
                       "（距下一阶段还差%s）" % json.dumps(stg["earned_next"], ensure_ascii=False)
                       if stg.get("earned_next") else ""),
            "当前特质条：" + self.traits.bars(),
        ]
        if now_lines:
            parts.append("性格现状：" + "；".join(now_lines))
        parts.append("经历塑造现状：" + stg["line"] + "；" + stance + "（信任 %.2f）" % tr)
        narrs = self.store.self_narrs()[:3]
        if narrs:
            parts.append("自我认知：" + "；".join(r[1] for r in narrs))
        return "\n".join(parts)

    # ------------------------------------------------ 视图（兼容旧 Persona）
    def md_path(self):
        dbp = getattr(self.store, "path", ":memory:")
        if dbp in (None, "", ":memory:"):
            return None
        return os.path.join(os.path.dirname(os.path.abspath(dbp)), "persona.md")

    def write_md(self, fitted_t=None):
        path = self.md_path()
        if path is None:
            return None
        stg = self.stage_info()
        t = self.traits.get()
        seed = self.traits.seed()
        lines = [
            "# 小星的成长快照", "",
            "> 真值在 calib 表 scope='growth'（只插不改）；本文件是人可读视图。",
            "> 每次塑形插一行 → 这份文件就是成长曲线的最新一帧。",
            "",
            "## 出厂种子（基因）",
            "", config.PERSONA,
            "种子特质：" + "  ".join("%s=%+.2f" % (TRAIT_LABEL[k], seed[k])
                                    for k in TRAIT_NAMES),
            "",
            "## 现在（%s期）" % stg["name"],
            "",
            "- 特质条：" + self.traits.bars(),
            "- 信任：%.2f" % self.trust(),
            "- 阶段限制：%s" % stg["line"],
        ]
        if stg.get("earned_next"):
            lines.append("- 距下一阶段：%s" % json.dumps(stg["earned_next"], ensure_ascii=False))
        narrs = self.store.self_narrs()
        lines += ["", "## 自我叙事（我是谁）", ""]
        # 先物化成列表再取 or：生成器对象恒真，挂 in-place 的 or 兜底永远
        # 不生效（审查 P3：空叙事时「（还没有）」占位行从未出现）
        narr_lines = ["- %s（权重 %.1f）" % (r[1], r[2]) for r in narrs[:8]]
        lines += narr_lines or ["- （还没有）"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def snapshot(self):
        """API 视图（/api/persona）。mu_now = 此刻完整底色（性格+昼夜），
        mu = 纯性格基线——前端画「出厂→现在→此刻」三点用。"""
        stg = self.stage_info()
        affect = getattr(self.dh, "affect", None)
        mu_now = None
        if affect is not None and hasattr(affect, "mu_at"):
            mu_now = [round(float(x), 3) for x in affect.mu_at()]
        return dict(mu=[round(float(x), 3) for x in self.mu()],
                    mu_now=mu_now, trust=self.trust(),
                    traits=self.traits.get(), seed=self.traits.seed(),
                    stage=stg["name"], stage_line=stg["line"],
                    next=stg.get("earned_next"),
                    narrs=[r[1] for r in self.store.self_narrs()],
                    last_shaped=self.store.get_meta("growth_shaped_at"),
                    md=self.md_path())
