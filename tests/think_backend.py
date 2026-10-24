#!/usr/bin/env python3
"""Mock thinking backend: first chunk carries delta.reasoning (Qwen-style)."""
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()
PORT = int(os.environ.get("PORT", "18281"))


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = json.loads(await request.body())
    if body.get("stream"):
        async def gen():
            # llama.cpp Qwen thinking: role chunk first, then reasoning chunks.
            yield 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}\n\n'
            yield 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"reasoning":"Let me check the documentation I already fetched earlier."}}]}\n\n'
            yield 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"reasoning":"From the README I already have:"}}]}\n\n'
            yield 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{"content":"So yes, --xmltv-url can be an address."},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"qwen","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
