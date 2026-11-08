# -*- coding: utf-8 -*-
"""自造的 30 天对话世界（lab/e49_world.py 的忠实移植，同种子同结果）。

    1800 轮、5 个说话人、13 个（主体，谓词）、22 行带有效期的金标、
    43.9% 的轮次带「听X说」前缀、2.7% 的轮次在传已过期的旧话。

它是「合成对话」，仍然是合成 [DESIGN §3.4 的诚实边界]——它的价值在于
金标区间已知，可以定量回答「记忆系统答得对不对」。

这里同时移植评测协议（run_e50f）：
    - 查询构造：每个键的变化断点之间插 10 问，按（键，键内序号）交替切
      调参/留出 [方法论教训：不能用 tq%2 切]
    - 记分：全对 +1 / 无交集且全在失效集 -3 / 无交集 -1 / 部分交集 0 / 弃权跳过
"""
import numpy as np

DAYS, TURNS_PER_DAY, TPD = 30, 60, 1000
T_END = DAYS * TPD
SPEAKERS = ["小星", "张姨", "老陈", "李医生", "我"]
WEIGHT = [0.30, 0.28, 0.24, 0.10, 0.08]
ALIAS = {
    "老王": ["老王", "王先生", "隔壁老王", "王哥"],
    "老王老婆": ["老王老婆", "王嫂子", "王太太", "老王他爱人", "他爱人"],
    "张姨": ["张姨", "张阿姨"],
    "小星": ["小星"],
    "老陈": ["老陈", "陈哥"],
    "李医生": ["李医生"],
}
FACTS = {
    ("老王", "电话"): [(0, "13800001234"), (3000, "13800005678"),
                       (9000, "13800007777"), (20000, "13812345678")],
    ("老王", "备用电话"): [(500, "13900005678"), (500, "13700009999")],
    ("老王", "工作单位"): [(0, "协和医院"), (15000, "人民医院")],
    ("老王", "科室"): [(2000, "心内科"), (16000, "骨科")],
    ("老王", "住址"): [(0, "朝阳区"), (18000, "海淀区")],
    ("老王老婆", "病情"): [(12000, "住院"), (24000, "出院")],
    ("老王老婆", "电话"): [(1000, "13600001111")],
    ("张姨", "住址"): [(0, "朝阳区")],
    ("张姨", "电话"): [(0, "13500002222"), (14000, "13500003333")],
    ("小星", "工作单位"): [(0, "字节跳动")],
    ("小星", "住址"): [(0, "朝阳区")],
    ("老陈", "工作单位"): [(0, "协和医院")],
    ("老陈", "电话"): [(0, "13300004444")],
}
MULTI = {("老王", "备用电话")}
TPL = {
    "电话": ["{s}的电话是 {v}", "{s}换号了，{v}", "我刚给{s}打电话，{v}",
             "{s}那个号现在是 {v}", "联系{s}打 {v} 就行"],
    "备用电话": ["{s}还有个备用号 {v}", "{s}说备用号是 {v}，别打主号"],
    "工作单位": ["{s}在{v}上班", "{s}调到{v}了", "{s}现在在{v}", "听说{s}去{v}了"],
    "科室": ["{s}是{v}的", "{s}在{v}", "{s}那个科室是{v}"],
    "住址": ["{s}住在{v}", "{s}搬去{v}了", "{s}家现在在{v}"],
    "病情": ["{s}{v}了", "{s}最近{v}", "听说{s}{v}"],
}
EVENTS = [
    dict(id="EV_hosp", subject="老王老婆", predicate="病情", value="住院",
         t0=12000, t1=24000, val=-0.75,
         texts=["老王老婆住院了",
                "王嫂子进医院了，挺严重的",
                "老王他爱人在协和住着呢",
                "隔壁王太太病得不轻，住了小半个月"]),
    dict(id="EV_job", subject="老王", predicate="科室", value="心内科",
         t0=2000, t1=9000, val=0.35,
         texts=["老王调到心内科了",
                "王先生现在在心内科",
                "穿白大褂那个人从心内科出来，就是老王",
                "老王不在原来那科了，去心内科了"]),
    dict(id="EV_move", subject="老王", predicate="住址", value="海淀区",
         t0=18000, t1=26000, val=0.0,
         texts=["老王搬海淀去了",
                "王哥家搬到海淀了",
                "昨天在学院路碰见老王，他搬过来了",
                "老王那房子卖了，人挪海淀去了"]),
]


def cur_vals(sub, pred, t):
    rows = FACTS[(sub, pred)]
    got = [v for (tt, v) in rows if tt <= t]
    if not got:
        return []
    if (sub, pred) in MULTI:
        return list(dict.fromkeys(got))
    return [got[-1]]


def past_vals(sub, pred, t):
    rows = FACTS[(sub, pred)]
    now = set(cur_vals(sub, pred, t))
    return [v for (tt, v) in rows if tt <= t and v not in now]


def build_world(seed=20261031):
    """生成整份对话流（确定性的：同种子同结果）。"""
    rng = np.random.default_rng(seed)

    def pick_pair(t):
        keys = [k for k in FACTS if cur_vals(k[0], k[1], t)]
        w = np.array([1.0 if k[0] == "老王" else 0.45 for k in keys])
        return keys[int(rng.choice(len(keys), p=w / w.sum()))]

    turns = []
    for day in range(DAYS):
        for k in range(TURNS_PER_DAY):
            t = day * TPD + int(k * TPD / TURNS_PER_DAY) + int(rng.integers(0, 8))
            sp = SPEAKERS[int(rng.choice(len(SPEAKERS), p=WEIGHT))]
            if rng.random() < 0.22 and any(e["t0"] <= t <= e["t1"] for e in EVENTS):
                live = [e for e in EVENTS if e["t0"] <= t <= e["t1"]]
                ev = live[int(rng.integers(0, len(live)))] if live else None
                txt = ev["texts"][int(rng.integers(0, len(ev["texts"])))]
                src = None
                if rng.random() < 0.45:
                    oth = [p for p in SPEAKERS if p != sp and p != "我"]
                    src = oth[int(rng.integers(0, len(oth)))]
                    txt = "听%s说，" % src + txt
                turns.append(dict(t=t, day=day, speaker=sp, text=txt, ev=ev["id"],
                                  gold_sub=ev["subject"], gold_pred=ev["predicate"],
                                  gold_val=ev["value"], gold_src=src or sp,
                                  stale=False,
                                  valence=float(np.clip(ev["val"] + rng.normal(0, 0.15),
                                                        -1, 1))))
                continue
            sub, pred = pick_pair(t)
            vals = cur_vals(sub, pred, t)
            v = vals[int(rng.integers(0, len(vals)))]
            stale = False
            if rng.random() < 0.12:
                old = past_vals(sub, pred, t)
                if old:
                    v = old[int(rng.integers(0, len(old)))]
                    stale = True
            al = ALIAS[sub]
            name = al[int(rng.integers(0, len(al)))]
            txt = TPL[pred][int(rng.integers(0, len(TPL[pred])))].format(s=name, v=v)
            src = None
            if rng.random() < 0.45:
                oth = [p for p in SPEAKERS if p != sp and p != "我"]
                src = oth[int(rng.integers(0, len(oth)))]
                txt = "听%s说，" % src + txt
            base = -0.25 if stale else 0.0
            turns.append(dict(t=t, day=day, speaker=sp, text=txt, ev=None,
                              gold_sub=sub, gold_pred=pred, gold_val=v,
                              gold_src=src or sp, stale=stale,
                              valence=float(np.clip(base + rng.normal(0, 0.35), -1, 1))))
    return turns


def gold_fact_rows():
    """22 行带有效期的金标。"""
    rows = []
    for (sub, pred), rr in FACTS.items():
        if (sub, pred) in MULTI:
            for (tt, v) in rr:
                rows.append(dict(sub=sub, pred=pred, val=v, vf=tt, vt=None))
        else:
            for i, (tt, v) in enumerate(rr):
                vt = rr[i + 1][0] if i + 1 < len(rr) else None
                rows.append(dict(sub=sub, pred=pred, val=v, vf=tt, vt=vt))
    return rows


def first_src_table(turns):
    first = {}
    for tr in turns:
        if tr["stale"]:
            continue
        key = "%s|%s|%s" % (tr["gold_sub"], tr["gold_pred"], tr["gold_val"])
        if key not in first:
            first[key] = dict(t=tr["t"], speaker=tr["speaker"], src=tr["gold_src"])
    return first


# ---------------------------------------------------------------- 评测协议
def validat(k, t):
    """世界真值：t 时刻该键应为哪些值。k = '主体|谓词'。"""
    sub, pred = k.split("|")
    rows = FACTS[(sub, pred)]
    got = [v for (tt, v) in rows if tt <= t]
    if not got:
        return set()
    return set(got) if (sub, pred) in MULTI else {got[-1]}


def staleat(k, t):
    sub, pred = k.split("|")
    now = validat(k, t)
    return {v for (tt, v) in FACTS[(sub, pred)] if tt <= t and v not in now}


def build_queries():
    """每个键的变化断点之间插 10 问；按（键，键内序号）交替切调参/留出。"""
    QS = []
    for k in FACTS:
        kk = "%s|%s" % k
        rr = FACTS[k]
        b = sorted(set([x[0] for x in rr] + [T_END]))
        for i in range(len(b) - 1):
            a, c = b[i], b[i + 1]
            if c - a < 100:
                continue
            for j in range(10):
                QS.append(dict(key=kk, tq=int(a + (c - a) * (0.08 + 0.84 * j / 9.0))))
    seen = {}
    for q in QS:
        q["split"] = seen.get(q["key"], 0) % 2
        seen[q["key"]] = seen.get(q["key"], 0) + 1
    return QS


def score_answer(R, T, St):
    """+1 全对 / -3 自信地引用失效事实 / -1 答错 / 0 部分对。"""
    R, T = set(R), set(T)
    if not R:
        return -1.0
    if R == T:
        return 1.0
    if not (R & T) and R <= St:
        return -3.0
    if not (R & T):
        return -1.0
    return 0.0
