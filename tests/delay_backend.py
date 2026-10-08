#!/usr/bin/env python3
"""Mock backend that stays silent (prefill) before its first token.

Used by e2e_ttft.py to prove the proxy measures TTFT to the first REAL
token (not the moment the SSE stream opens)."""
import asyncio
import json
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
PORT = int(os.environ.get("PORT", "18181"))
DELAY = float(os.environ.get("DELAY", "2.0"))


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = json.loads(await request.body())
    if body.get("stream"):
        async def gen():
            # Simulate prefill / queue: no byte of payload at all until
            # the first generated token (server keeps the connection open).
            await asyncio.sleep(DELAY)
            yield 'data: {"id": "b1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "premier token"}}]}\n\n'
            yield 'data: {"id": "b2", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}]}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
