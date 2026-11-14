# -*- coding: utf-8 -*-
"""抽取器：一句话 -> (来源, 主体, 谓词, 值, 实体表, 情绪)。

两个实测修正都在这里 [GRAPH §10.1 / SUMMARY 端到端]：
    1. 「听X说，」前缀必须剥掉，X 是**来源**（边上的一列属性），不是主体。
       朴素做法 790/790 轮把传话人当主角（张冠李戴），剥掉后留出效用 +0.2692。
    2. 来源要留**最早**那条（谁最早说的），不是最近一次说话人。
       存来源取最早 = 1.000，不存 = 0.409，取最近 = 0.455。

抽取器是正则 + 别名表，不是模型；「他」「那个谁」这类指代一个都没测
[GRAPH §10.4] —— 这是诚实边界，不是遗漏。
"""
import re

# ---------------------------------------------------------------- 别名
DEFAULT_ALIAS = {
    "老王": ["老王", "王先生", "隔壁老王", "王哥"],
    "老王老婆": ["老王老婆", "王嫂子", "王太太", "老王他爱人", "他爱人"],
    "张姨": ["张姨", "张阿姨"],
    "小星": ["小星"],
    "老陈": ["老陈", "陈哥"],
    "李医生": ["李医生"],
}

_PHONE = re.compile(r"1\d{10}")
_SRC = re.compile(r"^听(.+?)说，")
_UNITS = ["协和医院", "人民医院", "字节跳动"]
_DEPTS = ["心内科", "骨科"]
Districts = ["朝阳区", "海淀区"]
# 病情：模式 -> 归一值（「进医院」「住了院」都说的是住院）
_SICK = [("住院", "住院"), ("出院", "出院"), ("进医院", "住院")]

# 明确的变更信号：出现才允许「复活」一个已失效的旧值
_CHANGE_WORDS = ("换号", "换了", "改成", "改用", "新号", "搬到", "调到", "搬去")

# 问句里的谓词线索（问句不含值，谓词只能从名词推）。
# 每项 = (必须全部出现的词, 任一出现的词, 谓词)；顺序即优先级。
_QUESTION_PRED = [
    (("备用",), ("电话", "号"), "备用电话"),
    ((), ("电话", "号码"), "电话"),
    ((), ("单位", "上班", "公司", "工作"), "工作单位"),
    ((), ("科室", "内科", "骨科"), "科室"),
    # 「住」单字太贪（住院/住着都中招），用词组
    ((), ("住在", "搬", "住址", "家在"), "住址"),
    ((), ("病", "病情", "身体", "怎么样", "如何"), "病情"),
]

# 效价小词典（离线兜底口径；真实系统用 DigitalHuman(valence_fn=…) 换成评价器）。
# 否定翻转：「不开心」=负、「不烦」=正——词前紧邻否定词时符号取反。
_NEGATORS = ("不", "没", "无", "别", "非", "未")
_POS_WORDS = ["出院", "新车", "升", "好事", "开心", "高兴", "顺利", "康复",
              "恢复", "成功", "喜欢", "满意", "幸福", "放心", "健康",
              "棒", "厉害", "聪明", "可爱", "爱你", "谢谢", "帅", "好看"]
_NEG_WORDS = ["住院", "吵架", "病", "不好", "难", "烦", "去世", "糟", "难过",
              "伤心", "痛苦", "糟糕", "担心", "紧张", "生气", "失败", "累",
              # 骂人/攻击类——挨骂要能被感知到（委屈/生气都从这进）
              "笨", "蠢", "傻", "憨", "滚", "废物", "白痴", "笨蛋", "蠢货",
              "傻子", "憨子", "混蛋", "坏蛋", "讨厌", "恶心", "差劲", "闭嘴",
              "凶", "骂", "无语", "滚开", "闭口"]

# ---------------------------------------------------------------- 实体关系
# 亲属/社会关系称呼词，全部归一到规范形（谓词只有一种写法）。
# 关系是「from 对 to 的称呼」：「小王是小李他哥」= 小李管小王叫哥哥
# = (小李, 关系.哥哥, 小王)
_KIN_CANON = {"哥": "哥哥", "弟": "弟弟", "姐": "姐姐", "妹": "妹妹",
              "爸": "父亲", "妈": "母亲", "爸爸": "父亲", "妈妈": "母亲",
              "丈夫": "老公", "妻子": "老婆", "老爷子": "父亲", "老太太": "母亲",
              "师傅": "师父"}
_KIN_FULL = ["哥哥", "弟弟", "姐姐", "妹妹", "父亲", "母亲", "儿子", "女儿",
             "老公", "老婆", "男朋友", "女朋友", "师父", "徒弟", "老师", "学生",
             "上司", "下属", "老板", "员工",
             # 社会关系：非互斥（多值并存），不进 REL_CONFLICT
             "朋友", "同学", "同事", "邻居", "室友", "闺蜜", "死党", "网友",
             "战友", "亲戚", "搭档", "前男友", "前女友", "合伙人"]
_KIN_ALL = sorted(set(_KIN_FULL) | set(_KIN_CANON), key=len, reverse=True)
_KIN_RE = re.compile("|".join(_KIN_ALL))


def has_kinship_term(text):
    """句中是否出现亲属/社会关系称呼（我姐姐/他师父/我朋友…）。

    这类句子的「我」不指属主本人——「我姐姐的电话」不是「我的电话」，
    问句路由不得把主语消解到属主（错答比不答更糟 [context 同款教训]）。"""
    return bool(_KIN_RE.search(text or ""))

# 同一对人之间只能成立一个的关系组：说「弟弟」时，「哥哥/姐姐」自动撤回
# （互斥是关系本身的语义，不是多值并存——记错关系=换关系）。
REL_CONFLICT = {
    "哥哥": {"弟弟", "妹妹", "姐姐"},
    "弟弟": {"哥哥", "姐姐", "妹妹"},
    "姐姐": {"弟弟", "妹妹", "哥哥"},
    "妹妹": {"姐姐", "哥哥", "弟弟"},
    "父亲": {"儿子", "女儿"},
    "母亲": {"儿子", "女儿"},
    "儿子": {"父亲", "母亲"},
    "女儿": {"父亲", "母亲"},
    "老公": {"老婆"}, "老婆": {"老公"},
    "男朋友": {"女朋友"}, "女朋友": {"男朋友"},
    "师父": {"徒弟"}, "徒弟": {"师父"},
    "老师": {"学生"}, "学生": {"老师"},
    "上司": {"下属", "员工"}, "下属": {"上司", "老板"},
    "老板": {"员工", "下属"}, "员工": {"老板", "上司"},
}
_PRONOUN = {"我", "你", "他", "她", "它", "咱", "大家", "自己"}
# 名字里不该出现的字：防「告诉过我小王」这类把整句话截成名字的误匹配
_NAME_BLOCK = "的我你他她它您说"
_NAME_RE = r"[\u4e00-\u9fff0-9A-Za-z]{1,6}"
# 「X是Y的R」「X的R是Y」「X是Y他R（口语短形）」
_REL_PATTERNS = [
    re.compile(r"(%s)是(%s)的(%s)" % (_NAME_RE, _NAME_RE, "|".join(_KIN_ALL))),
    re.compile(r"(%s)的(%s)是(%s)" % (_NAME_RE, "|".join(_KIN_ALL), _NAME_RE)),
    re.compile(r"(%s)是(%s)他(%s)" % (_NAME_RE, _NAME_RE, "|".join(_KIN_CANON))),
]


def detect_relations(body):
    """从一句话里抽「谁是谁的什么」——返回 [(from, rel, to)]（规范关系词）。

    离线确定性（Tier 0）：只认称呼词表里的关系、正文逐字出现的名字；
    代词不算名字（指代消解是独立缺口，宁可不抽）。
    """
    out = []
    for i, pat in enumerate(_REL_PATTERNS):
        for m in pat.finditer(body):
            if i == 1:                           # 「X的R是Y」：第 2 组才是关系词
                frm, r, to = m.group(1), m.group(2), m.group(3)
            else:                                # 「X是Y(的/他)R」：Y 管一个叫 R
                a, b, r = m.group(1), m.group(2), m.group(3)
                frm, to = b, a
            r = _KIN_CANON.get(r, r)
            if (frm in _PRONOUN or to in _PRONOUN or frm == to
                    or any(c in _NAME_BLOCK for c in frm + to)):
                continue
            out.append((frm, r, to))
    return out


class Extractor:
    def __init__(self, alias=None):
        self.alias = dict(DEFAULT_ALIAS if alias is None else alias)
        # 最长匹配优先
        self._names = sorted(((nm, k) for k, al in self.alias.items() for nm in al),
                             key=lambda x: -len(x[0]))

    def canon(self, name):
        for k, al in self.alias.items():
            if name in al:
                return k
        return None

    def learn_alias(self, name, canon=None):
        """学到新实体别名（LLM 抽取回写）；返回是否真的新增。
        新实体（canon 不在表里）会作为新键登记。"""
        canon = canon or name
        if not name or name in self._all_names():
            return False
        if canon in self.alias:
            if name in self.alias[canon]:
                return False
            self.alias[canon].append(name)
        else:
            self.alias[canon] = list(dict.fromkeys([canon, name]))
        self._names = sorted(((nm, k) for k, al in self.alias.items() for nm in al),
                             key=lambda x: -len(x[0]))
        return True

    def _all_names(self):
        return {nm for al in self.alias.values() for nm in al}

    # ------------------------------------------------ 实体
    def find_entities(self, body):
        """别名最长匹配，位置不重叠（run_e50f.find_all 的忠实移植）。"""
        out, used = [], [False] * (len(body) + 1)
        for nm, ent in self._names:
            st = 0
            while True:
                i = body.find(nm, st)
                if i < 0:
                    break
                if not any(used[i:i + len(nm)]):
                    out.append((i, nm, ent))
                    for j in range(i, i + len(nm)):
                        used[j] = True
                st = i + 1
        out.sort()
        return out

    # ------------------------------------------------ 来源
    def split_source(self, text):
        """剥「听X说，」前缀。返回 (来源实体或None, 剩下的正文)。"""
        m = _SRC.match(text)
        if not m:
            return None, text
        return self.canon(m.group(1)), text[m.end():]

    # ------------------------------------------------ 谓词与值
    @staticmethod
    def predicate_value(body):
        m = _PHONE.search(body)
        if m:
            return ("备用电话" if "备用" in body else "电话"), m.group(0)
        for u in _UNITS:
            if u in body:
                return "工作单位", u
        for d in _DEPTS:
            if d in body:
                return "科室", d
        for d in Districts:
            if d in body:
                return "住址", d
        for pat, val in _SICK:
            i = body.find(pat)
            if i >= 0:
                # 否定语境不算断言：「没进医院」不能记成住院（宁可漏记）
                if _negated_at(body, i):
                    continue
                return "病情", val
        return None, None

    @staticmethod
    def signals_change(text):
        return any(w in text for w in _CHANGE_WORDS)

    def parse_question(self, text):
        """问句：谓词从名词推，值恒为 None。返回 dict 或 None。"""
        src, body = self.split_source(text)
        ents = self.find_entities(body)
        if not ents:
            return None
        pred = None
        for must, anyof, p in _QUESTION_PRED:
            if all(w in body for w in must) and any(w in body for w in anyof):
                pred = p
                break
        if pred is None:
            return None
        return dict(src=src, subject=ents[0][2], predicate=pred, value=None,
                    entities=ents, change=False, body=body)

    # ------------------------------------------------ 情绪
    @staticmethod
    def valence(text):
        """小词典效价 ±1/0：正负词计数相减；词前紧邻否定词则翻转该词符号
        （「不开心」计负、「不烦」计正）。双否定（「不是不开心」）不处理——
        小词典的诚实边界，真实评价器走 DigitalHuman(valence_fn=…) 注入。"""
        score = 0
        for words, sign in ((_POS_WORDS, 1), (_NEG_WORDS, -1)):
            for w in words:
                i = text.find(w)
                while i >= 0:
                    if text[max(0, i - 2):i].endswith(_NEGATORS):
                        score -= sign
                    else:
                        score += sign
                    i = text.find(w, i + 1)
        return float(np_sign(score))

    # ------------------------------------------------ 一站式
    def parse(self, text, speaker=None):
        """返回 dict 或 None（抽不出事实时）。

        src    传话人（规范名或 None）
        subject 主体（正文里第一个实体——剥完前缀之后的）
        predicate/value
        entities  正文全部实体 [(pos, surface, canon)]
        change   是否带变更信号
        """
        src, body = self.split_source(text)
        ents = self.find_entities(body)
        if not ents:
            return None
        pred, val = self.predicate_value(body)
        if pred is None:
            return None
        return dict(src=src, subject=ents[0][2], predicate=pred,
                    value=val, entities=ents, change=self.signals_change(text),
                    body=body)


def np_sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _negated_at(text, i):
    """位置 i 之前的紧邻两字是否含否定词（「没进医院」不是住院）。"""
    return text[max(0, i - 2):i].endswith(_NEGATORS)


def normalize_value(predicate, value):
    """值的确定性归一（Tier 0 词典口径，公开给写入侧等价裁决用）。

    Tier 0 的 parse() 走 predicate_value 时已经归一（进医院→住院）；
    LLM Tier 1 的值逐字落地（G3），同一归一它没有——这里补上同一张表，
    让两条抽取路对同一个谓词产出同一个规范值。其余谓词原样返回。
    否定语境不归一：「根本没进医院」里命中「进医院」，但它的意思不是
    住院——归一了就会经等价裁决把已记的「住院」confirm 一遍。"""
    if predicate == "病情":
        for pat, val in _SICK:
            i = value.find(pat)
            if i >= 0:
                return value if _negated_at(value, i) else val
    return value


def is_segment_boundary(t, prev_t, gap_s=None):
    """段由 >30 min 空隙切开 [SUMMARY §6.5]。"""
    gap_s = gap_s if gap_s is not None else 1800.0
    return prev_t is None or (t - prev_t) > gap_s


# ---------------------------------------------------------------- 属主自报家门
# 「我」的消解锚（meta.owner）没人登记时，writer 只能把「我」的陈述整句拒写
# （宁可漏记）。自报家门是唯一安全的登记口：整句就是 intro，不是顺嘴一说。
_SELF_INTRO_RE = re.compile(
    r"^我(?:叫|的名字是|是)([\u4e00-\u9fa50-9A-Za-z]{2,6})[，,。！!？?\s]*$")
_OWNER_BAD_FIRST = set("说想要在来去没不是很太个这那怎为和跟对被把就也")


def self_intro_name(text):
    """「我叫三上」/「我是三上」-> 候选属主名；整句自报家门才算，防误伤。"""
    m = _SELF_INTRO_RE.match(text.strip())
    if not m:
        return None
    name = m.group(1)
    if name[0] in _OWNER_BAD_FIRST or name in _PRONOUN or name in _KIN_ALL:
        return None
    return name
