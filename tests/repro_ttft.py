"""Repro: routing block must close on a thinking-model stream."""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

PORT = 18449
BASE = f"http://127.0.0.1:{PORT}"

def start(*args, log, **env):
    return subprocess.Popen(
        [sys.executable, *args],
        env={**os.environ, **env},
        cwd=os.getcwd(),
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )

backend = start("tests/think_backend.py", log="/tmp/think.log", PORT="18281")
proxy = start(
    "-m", "uvicorn", "proxy:app", "--port", str(PORT), "--log-level", "debug",
    log="/tmp/proxy_think.log",
    LLAMA_BACKEND="http://127.0.0.1:18281",
    DEFAULT_REASONING_EFFORT="low",
)
try:
    ok = False
    for _ in range(60):
        try:
            urllib.request.urlopen(BASE + "/proxy-health", timeout=2); ok = True; break
        except Exception:
            time.sleep(0.5)
    assert ok, "proxy not up"

    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=60)
    events = []
    buf = b""
    while True:
        buf += resp.read(1)
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            events.append(ev.decode(errors="replace"))
        if b"[DONE]" in buf:
            break

    text = "\n".join(events)
    has_block = "CHOIX DU ROUTEUR" in text
    has_ttft = "TTFT:" in text
    has_fin = "===== FIN =====" in text
    print(f"block_open={has_block}  ttft={has_ttft}  fin={has_fin}")
    for i, ev in enumerate(events):
        if "TTFT" in ev:
            m = re.search(r"TTFT: ([\d.]+) s", ev)
            print(f"event {i}: {m.group(0) if m else ev[:80]}")
    assert has_block, "no routing block"
    assert has_ttft, "TTFT missing — block never closed with a TTFT"
    assert has_fin, "===== FIN ===== missing"
    print("OK")
finally:
    for p in (proxy, backend):
        p.kill()
