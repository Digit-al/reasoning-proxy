#!/usr/bin/env python3
"""Mock sidecar LLM (OpenAI-compatible) for proxy testing.

Answers the classifier JSON: {"task": "...", "effort": "..."}.
The last user message drives the choice:
- contains "creative"/"poem"/"story"/"haiku"  -> {"task":"creative","effort":"low"}
- contains "hard"/"proof"/"debug"/"quantum"   -> {"task":"reasoning","effort":"xhigh"}
- otherwise                                   -> {"task":"reasoning","effort":"medium"}
"""
import json
import os

from fastapi import FastAPI, Request

app = FastAPI()
CALLS: list[dict] = []
ANSWER = os.environ.get("SIDECAR_ANSWER", "")  # override: raw answer text


def _classify(text: str) -> str:
    """Classify based on the USER's message only (strip the trailing JSON
    format hint so "creative" in the format does not leak into the
    creative branch)."""
    # Extract the USER: line (or first line) to classify.
    user_line = ""
    for line in text.splitlines():
        if line.startswith("USER:"):
            user_line = line[5:].strip()
            break
    t = user_line.lower() if user_line else text.lower()
    if any(w in t for w in ("poem", "haiku", "story", "poème", "saga")):
        return '{"task": "creative", "effort": "low"}'
    if any(w in t for w in ("hard", "proof", "debug", "quantum", "xhigh")):
        return '{"task": "reasoning", "effort": "xhigh"}'
    return '{"task": "reasoning", "effort": "medium"}'


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = json.loads(await request.body())
    CALLS.append(body)
    last_user = ""
    for m in body.get("messages", []):
        if m.get("role") == "user":
            last_user = m.get("content", "")
    if ANSWER:
        answer = ANSWER
    else:
        answer = _classify(last_user)
    return {
        "id": "side",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@app.get("/calls")
async def calls():
    return {"count": len(CALLS), "last": CALLS[-1] if CALLS else None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "18082")), log_level="warning")
