#!/usr/bin/env python3
"""Mock llama.cpp OpenAI-compatible backend for proxy testing."""
import json
import os
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
RECEIVED: list[dict] = []
PORT = int(os.environ.get("PORT", "18081"))
MODEL_ID = os.environ.get("MODEL_ID", "mock-model")


def _sse(payload: list[dict]) -> "StreamingResponse":
    async def gen():
        for chunk in payload:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = json.loads(await request.body())
    RECEIVED.append(body)
    effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
    if body.get("stream"):
        return _sse(
            [
                {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": f"[echo effort={effort}] "}}]},
                {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
                {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
        )
    return {
        "id": "x",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": f"answer effort={effort}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


@app.get("/v1/models")
async def models():
    return {"data": [{"id": MODEL_ID, "object": "model"}]}


@app.get("/models")
async def models_mgmt():
    """llama.cpp root model-management list (no status on model 1,
    explicit status on model 2 — must not be overwritten)."""
    return {
        "data": [
            {"model": MODEL_ID, "size": 12345},
            {"model": MODEL_ID + "-2", "status": {"value": "unloaded", "context": "unloaded by user"}},
        ]
    }


@app.get("/received")
async def received():
    return {"bodies": RECEIVED}


@app.post("/v1/echo")
async def echo(request: Request):
    """Return whatever body was sent, to verify byte-for-byte passthrough."""
    return {"got": json.loads(await request.body())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
