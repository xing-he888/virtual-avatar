# -*- coding: utf-8 -*-
"""dh.growth：成长层——性格的慢变量源（用户口中的「左脑性格层」）。

分工：brain（主图）决定做什么，limbic（右脑）决定带着什么劲做，
growth 决定「他是一个什么样的人、正在变成什么样的人」——向右脑供
情绪基线（OCEAN→Gebhard 公式→mu），向主图供人格描述与特质偏置，
向记忆供塑形证据的权重口径。

三文件：
    traits.py   OCEAN 五维真值（calib 只插不改）+ Gebhard/ALMA 回归 → PAD
    stages.py   发展阶段（初生→幼生→成长→成熟，只进不退，调制学习率/游移/话长）
    persona.py  塑形引擎（归因规则表 + salience 加权 + EMA + 钳制）
                + 三个塑造接口（自主 shape / 对话式教养 nurture / 编程 shape_direct）
                + 动态人格描述（替写死的出厂字符串）
"""
from .persona import Persona
from .traits import Traits, TRAIT_NAMES, TRAIT_LABEL

__all__ = ["Persona", "Traits", "TRAIT_NAMES", "TRAIT_LABEL"]
