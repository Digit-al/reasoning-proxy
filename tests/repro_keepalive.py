"""Repro: first backend chunk has NO 'id' field (some backends omit it).
The routing block must still close (treat as a real token)."""
import json, os, re, subprocess, sys, time, urllib.request

PORT = 18551
BASE = f"http://127.0.0.1:{PORT}"

def start(*args, log, **env):
    return subprocess.Popen([sys.executable, *args], env={**os.environ, **env},
                            cwd=os.getcwd(), stdout=open(log, "w"), stderr=subprocess.STDOUT)

# backend that sends keepalives + a first chunk with NO 'id' field
backend_code = r'''
import json, os, time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
app = FastAPI()
PORT = int(os.environ.get("PORT", "18552"))
@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = json.loads(await request.body())
    async def gen():
        # keepalives during prefill
        yield ": keepalive\n\n"
        yield ": keepalive\n\n"
        # first real chunk WITHOUT an 'id' field
        yield 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"bonjour"}}]}\n\n'
        yield 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"salut"}}]}\n\n'
        yield "data: [DONE]\n\n"
    if body.get("stream"):
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"choices":[{"message":{"role":"assistant","content":"ok"}}]}
'''
with open("tests/_keepalive_backend.py", "w") as f:
    f.write(backend_code)

backend = start("tests/_keepalive_backend.py", log="/tmp/kb.log", PORT="18552")
proxy = start("-m", "uvicorn", "proxy:app", "--port", str(PORT), "--log-level", "debug",
              log="/tmp/proxy_kb.log", LLAMA_BACKEND="http://127.0.0.1:18552",
              DEFAULT_REASONING_EFFORT="low")
try:
    ok = False
    for _ in range(60):
        try:
            urllib.request.urlopen(BASE + "/proxy-health", timeout=2); ok = True; break
        except Exception:
            time.sleep(0.5)
    assert ok, "proxy not up"
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps({"messages":[{"role":"user","content":"hi"}],"stream":True}).encode(),
        headers={"Content-Type":"application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    events, buf = [], b""
    while True:
        buf += resp.read(1)
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1); events.append(ev.decode(errors="replace"))
        if b"[DONE]" in buf: break
    text = "\n".join(events)
    print(f"block={'CHOIX DU ROUTEUR' in text}  ttft={'TTFT:' in text}  fin={'===== FIN =====' in text}")
    m = re.search(r"TTFT: ([\d.]+) s", text)
    print("TTFT value:", m.group(0) if m else "MISSING")
    assert "TTFT:" in text, "block never closed with TTFT"
    assert "===== FIN =====" in text, "block never closed"
    print("OK")
finally:
    for p in (proxy, backend): p.kill()
