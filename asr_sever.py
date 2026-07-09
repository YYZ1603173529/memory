#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-ASR-1.7B 本地 ASR 服务（OpenAI 兼容 API）
端口: 8001
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch, base64, uvicorn, tempfile, os, logging
from qwen_asr import Qwen3ASRModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr_server")

app = FastAPI()

# ---------- 模型加载 ----------
MODEL_PATH = os.environ.get("ASR_MODEL_PATH", "/home/chenyifu/YYZ/Qwen3-ASR-1.7B")
DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")

logger.info(f"Loading Qwen3-ASR-1.7B from {MODEL_PATH} on {DEVICE} ...")
model = Qwen3ASRModel.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map=DEVICE,
    max_inference_batch_size=32,
    max_new_tokens=256,
)
logger.info("Qwen3-ASR-1.7B loaded successfully.")


# ---------- 请求模型 ----------
class AudioPart(BaseModel):
    type: str
    input_audio: dict

class Message(BaseModel):
    role: str
    content: list[dict]

class Request(BaseModel):
    model: str = "qwen3-asr"
    messages: list[Message]


# ---------- API 端点 ----------
@app.post("/v1/chat/completions")
async def chat(req: Request):
    for msg in req.messages:
        for part in msg.content:
            if part.get("type") == "input_audio":
                data = part.get("input_audio", {}).get("data", "")
                if not data:
                    continue

                # 去掉 data URL 前缀 (data:audio/wav;base64,)
                if "," in data:
                    data = data.split(",", 1)[1]

                audio_bytes = base64.b64decode(data)
                logger.info(f"Received audio: {len(audio_bytes)} bytes")

                # 写入临时文件供 qwen-asr 使用
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_bytes)
                    tmp_path = f.name

                try:
                    results = model.transcribe(audio=tmp_path, language=None)
                    text = results[0].text if results else ""
                    logger.info(f"ASR result: {text[:100]}...")
                except Exception as e:
                    logger.error(f"ASR inference error: {e}")
                    text = ""
                finally:
                    os.unlink(tmp_path)

                return {
                    "id": "asr-1",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text or ""}}],
                }

    return {
        "id": "asr-0",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
    }


@app.get("/health")
async def health():
    return {"ok": True, "model": "Qwen3-ASR-1.7B"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
