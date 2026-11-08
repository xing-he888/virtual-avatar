from pydantic import BaseModel
from enum import Enum
from typing import Optional

'''
作为总体的配置
'''
class Gender(Enum):
    MALE='male'
    FAMALE='famale'
    OTHER='other'


class totalConfig(BaseModel):
    gender:Gender
    height:int
    vision:int
    age:int
    educationBackground:Optional[str]
    familyBackground:Optional[str]
    
