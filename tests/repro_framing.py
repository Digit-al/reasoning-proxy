"""Verify SSE framing when keepalives precede the first token (prod scenario).

Captures the EXACT bytes OWUI would receive and asserts that the
close-chunk (TTFT/FIN) is on its own SSE line (not glued to a keepalive
colon, which would turn it into a comment OWUI ignores)."""
import json, os, re, subprocess, sys, time, urllib.request

PORT = 18661
BASE = f"http://127.0.0.1:{PORT}"

backend_code = r'''
import json, os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
app = FastAPI()
PORT = int(os.environ.get("PORT", "18662"))
@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = json.loads(await request.body())
    async def gen():
        # bare-colon keepalives, exactly like the prod llama.cpp backend
        yield ":\n\n"
        yield ":\n\n"
        # first real token
        yield 'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"salut"}}]}\n\n'
        yield "data: [DONE]\n\n"
    if body.get("stream"):
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"choices":[{"message":{"role":"assistant","content":"ok"}}]}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
'''
with open("tests/_kb2.py", "w") as f:
    f.write(backend_code)

def start(*args, log, **env):
    return subprocess.Popen([sys.executable, *args], env={**os.environ, **env},
                            cwd=os.getcwd(), stdout=open(log, "w"), stderr=subprocess.STDOUT)

backend = start("tests/_kb2.py", log="/tmp/kb2.log", PORT="18662")
proxy = start("-m", "uvicorn", "proxy:app", "--port", str(PORT), "--log-level", "info",
              log="/tmp/proxy_kb2.log", LLAMA_BACKEND="http://127.0.0.1:18662",
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
    raw = b""
    while True:
        piece = resp.read(1024)
        if not piece:
            break
        raw += piece

    text = raw.decode(errors="replace")
    print("=== RAW BYTES OWUI RECEIVES ===")
    print(repr(raw[:600]))
    print("================================")

    # The close chunk (TTFT/FIN) must be a proper data: line, on its own.
    has_ttft = "TTFT:" in text
    has_fin = "===== FIN =====" in text
    print(f"TTFT present: {has_ttft}\nFIN present: {has_fin}")

    # CRITICAL: the close chunk's data: line must NOT be glued to a leading
    # colon (which would make OWUI treat the whole line as a comment).
    # Find the line containing 'TTFT:' and ensure it starts with 'data:'.
    ttft_line_ok = any(
        line.strip().startswith("data:") and "TTFT:" in line
        for line in text.split("\n")
    )
    print(f"TTFT on its own data: line: {ttft_line_ok}")

    assert has_ttft, "TTFT missing"
    assert has_fin, "FIN missing"
    assert ttft_line_ok, "TTFT chunk glued to a keepalive -> would be dropped by OWUI"
    print("OK - framing correct, OWUI will render the routing block close")
finally:
    for p in (proxy, backend):
        p.kill()
    import os as _o; _o.unlink("tests/_kb2.py")
