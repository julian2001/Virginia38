import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.runner import ModelRegistry

app = FastAPI(title="Vera VDR Model API", version="1.0")

# load registry
MODEL_NAME_DEFAULT = os.getenv("MODEL_NAME", "gpt5_biotech_demo")
registry = ModelRegistry(models_dir="models", default_model=MODEL_NAME_DEFAULT)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2

@app.get("/health")
def health():
    return {"ok": True, "active_model": registry.active_name}

@app.get("/models")
def list_models():
    return {"available": registry.list_models(), "active": registry.active_name}

@app.post("/models/load")
def load_model(payload: Dict[str, str]):
    name = payload.get("name")
    if not name:
        raise HTTPException(400, "Missing 'name'")
    try:
        registry.load(name)
        return {"loaded": name}
    except Exception as e:
        raise HTTPException(500, f"Load failed: {e}")

# Simple chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    model_name = req.model or registry.active_name
    try:
        model = registry.get(model_name)
        out = model.chat(
            messages=[m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature or 0.2,
        )
        return {"model": model_name, "choices": [{"message": {"role": "assistant", "content": out}}]}
    except Exception as e:
        raise HTTPException(500, f"Chat failed: {e}")

# OpenAI-compatible shim: /v1/chat/completions
@app.post("/v1/chat/completions")
def openai_chat(req: Dict[str, Any]):
    model_name = req.get("model") or registry.active_name
    messages = req.get("messages", [])
    temperature = req.get("temperature", 0.2)
    max_tokens = req.get("max_tokens", 512)
    try:
        model = registry.get(model_name)
        out = model.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
        return {
            "model": model_name,
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": out}, "finish_reason": "stop"}
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"OpenAI chat failed: {e}")
