"""
数字人的眼睛 —— 拍一帧，交给 deepseek 看，返回文字描述

用法:
    from vision.camera import look
    print(look("描述你看到什么"))
"""
import os
import base64

import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()   # 从当前目录向上自动查找 .env

client = OpenAI(
    api_key=os.environ.get("ALIYUN_APIKEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def look(question="这是你眼睛此刻看到的画面。请客观描述：画面里有什么人、什么物品、什么环境。", camera=0):
    """拍一张照片，问模型看到了什么，返回文字"""
    cap = cv2.VideoCapture(camera)

    # 前几帧是黑的（曝光没稳定），丢掉
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()

    if not ok:
        return "（摄像头打不开）"

    # 缩到 768px 宽，够看清又不浪费 token
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (768, int(768 * h / w)))
    ok, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode()

    r = client.chat.completions.create(
        model="deepseek-v4.1-flash",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": question},
        ]}],
        max_tokens=2000,   # 推理模型的思维链很长，给少了正式回答就是空的
    )
    return r.choices[0].message.content


if __name__ == "__main__":
    print(look())
