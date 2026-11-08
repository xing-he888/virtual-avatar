from pydantic import BaseModel
from base_config import totalConfig
'''
用于感知相关配置
'''
class virtualAwarenessConfig(BaseModel):
    linkingRating:int
    moodRating:int
