from pydantic import BaseModel,Field

class memoryConfig(BaseModel):
    memoryRating=Field(0.5,alias='记忆力指数')
    shortMemoryTime=Field(5,alias='短期记忆窗口年限')
    midMemoryTime=Field(10,alias='中期记忆窗口年限')
    longMemoryTime=Field(15,alias='长期记忆窗口年限')
