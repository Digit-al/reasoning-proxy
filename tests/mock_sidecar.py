#!/usr/bin/env python3
"""Mock sidecar LLM (OpenAI-compatible) for proxy testing."""
import json
import os

from fastapi import FastAPI, Request

app = FastAPI()
CALLS: list[dict] = []
ANSWER = os.environ.get("SIDECAR_ANSWER", "high")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = json.loads(await request.body())
    CALLS.append(body)
    # echo a bit of the prompt so the driver can assert what was sent
    last_user = ""
    for m in body.get("messages", []):
        if m.get("role") == "user":
            last_user = m.get("content", "")
    return {
        "id": "side",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "_echo": last_user[:200],
    }


@app.get("/calls")
async def calls():
    return {"count": len(CALLS), "last": CALLS[-1] if CALLS else None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "18082")), log_level="warning")
