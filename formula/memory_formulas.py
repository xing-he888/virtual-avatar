'''
数字人记忆遗忘模型 —— 艾宾浩斯遗忘曲线
主公式: R = e^(-t / S)
    R: 剩余记忆率 (0~1)
    t: 距上次接触/回忆经过的时间 (单位与 S 一致, 自定义: 天/小时均可)
    S: 记忆强度, 由 记忆档位 × 记忆力指数 × 心情 × 复习增益 共同决定
'''
import math
from enum import Enum


class MemoryLevel(Enum):
    '''记忆档位: 短期 / 中期 / 长期'''
    SHORT = 'short'
    MID = 'mid'
    LONG = 'long'


# 三档基准记忆强度 (比例 1:50:500, t 用什么单位 S 就是什么单位)
# 例: t 以"天"计 -> 短期约1天忘到37%, 中期50天, 长期500天
BASE_STRENGTH = {
    MemoryLevel.SHORT: 1.0,
    MemoryLevel.MID: 50.0,
    MemoryLevel.LONG: 500.0,
}

# 每被回忆一次, 记忆强度乘以该系数 (间隔重复的核心)
RECALL_BOOST = 1.5


def remaining_memory(
    elapsed: float,
    level: MemoryLevel = MemoryLevel.MID,
    memory_rating: float = 0.5,
    recall_count: int = 0,
    mood_factor: float = 1.0,
) -> float:
    '''
    计算一条记忆的剩余记忆率 R (0~1)

    :param elapsed:        距上次接触经过的时间 (单位自定, 需与基准强度同单位)
    :param level:          记忆档位 (短期/中期/长期)
    :param memory_rating:  记忆力指数 0~1, 对应 memoryConfig.memoryRating
    :param recall_count:   该记忆被主动回忆过的次数, 每次使 S × RECALL_BOOST
    :param mood_factor:    心情修正系数 (>1 记得更牢), 可由 moodRating 映射
    :return:               剩余记忆率, float 0~1
    '''
    s = BASE_STRENGTH[level]
    # 记忆力指数缩放: 0.5 为标准, 越高忘得越慢 (下限保护防除零/极端值)
    s *= max(memory_rating, 0.05) / 0.5
    # 心情修正
    s *= mood_factor
    # 复习增益
    s *= RECALL_BOOST ** recall_count
    return math.exp(-elapsed / s)


def is_forgotten(r: float, threshold: float = 0.15) -> bool:
    '''判断记忆是否已被遗忘 (R 低于阈值 -> 可降权/归档)'''
    return r < threshold

