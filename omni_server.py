#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-Omni-7B 本地推理服务（OpenAI 兼容 API）
端口: 8000

同时作为主模型（视频推理）和摘要模型（中期/长期记忆压缩）使用。
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch, base64, uvicorn, tempfile, os, logging, uuid, time
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni_server")

# ---------- 全局配置 ----------
MODEL_PATH = os.environ.get("OMNI_MODEL_PATH", "/home/chenyifu/YYZ/Qwen2.5-Omni-7B")

app = FastAPI()

# 模型和处理器在启动时加载
model = None
processor = None


def load_model():
    """延迟加载模型（避免 import 时加载）"""
    global model, processor
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    logger.info(f"Loading Qwen2.5-Omni-7B from {MODEL_PATH} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()  # 只输出文本，节省约2GB显存
    logger.info("Qwen2.5-Omni-7B loaded. talker disabled (text-only mode).")

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    logger.info("Processor loaded.")


# ---------- 辅助函数 ----------

def _decode_base64_image(data_url: str) -> str:
    """将 base64 data URL 解码并写入临时文件，返回文件路径"""
    if "," not in data_url:
        return None
    header, b64_data = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64_data)

    ext = ".jpg"
    if "png" in header:
        ext = ".png"
    elif "webp" in header:
        ext = ".webp"

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(img_bytes)
    tmp.close()
    return tmp.name


def _convert_openai_to_omni(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """
    将 OpenAI 格式的消息转换为 Qwen2.5-Omni conversation 格式。
    返回 (conversation, temp_file_paths)
    """
    conversation = []
    temp_files = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str):
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": content}],
                })
            elif isinstance(content, list):
                text_parts = [
                    item["text"] for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": "\n".join(text_parts)}],
                })
            continue

        if role == "assistant":
            if isinstance(content, str):
                conversation.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = [
                    item["text"] for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                conversation.append({"role": "assistant", "content": "\n".join(text_parts)})
            continue

        # role == "user"
        if isinstance(content, str):
            conversation.append({"role": "user", "content": content})
            continue

        if not isinstance(content, list):
            conversation.append({"role": "user", "content": str(content)})
            continue

        # user content is a list of parts
        new_content = []
        has_image = False
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")

            if item_type == "text":
                new_content.append({"type": "text", "text": item.get("text", "")})

            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = str(image_url)

                if url.startswith("data:"):
                    tmp_path = _decode_base64_image(url)
                    if tmp_path:
                        temp_files.append(tmp_path)
                        new_content.append({"type": "image", "image": tmp_path})
                        has_image = True
                elif url.startswith("file://"):
                    file_path = url[7:]
                    new_content.append({"type": "image", "image": file_path})
                    has_image = True
                elif os.path.exists(url):
                    new_content.append({"type": "image", "image": url})
                    has_image = True
                else:
                    logger.warning(f"Unsupported image URL: {url[:80]}...")

        if new_content:
            conversation.append({"role": "user", "content": new_content})

    return conversation, temp_files


# ---------- 请求模型 ----------

class Request(BaseModel):
    model: str = "qwen2.5-omni-7b"
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


# ---------- API 端点 ----------

@app.post("/v1/chat/completions")
async def chat(req: Request):
    global model, processor
    if model is None:
        load_model()

    temp_files = []
    try:
        # 转换消息格式
        conversation, temp_files = _convert_openai_to_omni(req.messages)

        # 检查是否有图片输入
        has_multimodal = any(
            isinstance(msg.get("content"), list) and
            any(item.get("type") == "image" for item in msg["content"] if isinstance(item, dict))
            for msg in conversation
        )

        # 准备推理
        try:
            from qwen_omni_utils import process_mm_info
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=False,
            )
        except ImportError:
            # 纯文本降级方案（不需要 qwen_omni_utils）
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(
                text=[text], return_tensors="pt", padding=True,
            )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            if has_multimodal and any(v is not None for v in [inputs.get("input_features"), inputs.get("pixel_values"), inputs.get("pixel_values_videos")]):
                text_ids = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature if req.temperature > 0 else 1.0,
                    top_p=req.top_p,
                    do_sample=req.temperature > 0,
                    return_audio=False,
                    use_audio_in_video=False,
                )
            else:
                text_ids = model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature if req.temperature > 0 else 1.0,
                    top_p=req.top_p,
                    do_sample=req.temperature > 0,
                    return_audio=False,
                )

        output_text = processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result_text = output_text[0].strip() if output_text else ""

        logger.info(
            f"Generated {len(result_text)} chars, "
            f"max_tokens={req.max_tokens}, temp={req.temperature}"
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    except Exception as e:
        logger.exception(f"Inference error: {e}")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "error",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    finally:
        # 清理临时文件
        for f in temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model": "Qwen2.5-Omni-7B",
        "loaded": model is not None,
    }


if __name__ == "__main__":
    # 启动时预加载模型
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
