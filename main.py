import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://host.docker.internal:12434/engines/llama.cpp/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "dmr-no-key-required")
MODEL_ID        = os.getenv("MODEL_ID", "ai/smollm2")

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

app = FastAPI(title="VDR+Bio API", version="0.1.0")

class ChatRequest(BaseModel):
    messages: list[dict]
    model: str | None = None
    temperature: float | None = 0.2
    max_tokens: int | None = 512

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "base_url": OPENAI_BASE_URL}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        model = req.model or MODEL_ID
        resp = client.chat.completions.create(
            model=model,
            messages=req.messages,
            temperature=req.temperature or 0.2,
            max_tokens=req.max_tokens or 512,
        )
        return {
            "model": model,
            "choices": [
                {
                    "message": {
                        "role": resp.choices[0].message.role,
                        "content": resp.choices[0].message.content,
                    }
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
