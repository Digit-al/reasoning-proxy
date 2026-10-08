"""E2E: TTFT must be measured to the first REAL token, not the stream open."""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

PORT = 18199
BASE = f"http://127.0.0.1:{PORT}"
DELAY = 2.0


def start(*args, cwd=None, **env):
    return subprocess.Popen(
        [sys.executable, *args],
        env={**os.environ, **env},
        cwd=cwd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def start(*args, cwd=None, **env):
    return subprocess.Popen(
        [sys.executable, *args],
        env={**os.environ, **env},
        cwd=cwd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def start_log(*args, log, **env):
    return subprocess.Popen(
        [sys.executable, *args],
        env={**os.environ, **env},
        cwd=os.getcwd(),
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )


_HERE = os.path.dirname(os.path.abspath(__file__))
delay_backend = start_log(
    os.path.join(_HERE, "delay_backend.py"), log="/tmp/db.log",
    PORT="18181", DELAY=str(DELAY),
)
proxy = start_log(
    "-m", "uvicorn", "proxy:app", "--port", str(PORT), "--log-level", "warning",
    log="/tmp/proxy.log",
    LLAMA_BACKEND="http://127.0.0.1:18181",
    DEFAULT_REASONING_EFFORT="medium",
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
        data=json.dumps({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=60)
    events = []
    buf = b""
    while True:
        buf += resp.read(1)
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            events.append(ev)
        if b"[DONE]" in buf:
            break

    t_ann = t_close = t_tok = None
    for i, ev in enumerate(events):
        text = ev.decode(errors="replace")
        if t_ann is None and "CHOIX DU ROUTEUR" in text:
            t_ann = i
        if t_close is None and "===== FIN =====" in text:
            t_close = i
        if t_tok is None and "premier token" in text:
            t_tok = i

    assert t_ann is not None and t_close is not None and t_tok is not None, (t_ann, t_close, t_tok)
    assert t_close < t_tok, "block must close BEFORE the first token"
    assert t_close == t_tok - 1, f"close must be immediately before token: {t_close} vs {t_tok}"

    ttft_reported = None
    for ev in events:
        text = ev.decode(errors="replace")
        m = re.search(r"TTFT: ([\d.]+) s", text)
        if m:
            ttft_reported = float(m.group(1))
    assert ttft_reported is not None, "TTFT line missing"
    # It must reflect the ~2s prefill delay, not ~0s (stream open).
    assert ttft_reported >= DELAY - 0.5, f"TTFT too low: {ttft_reported} (expected >= {DELAY-0.5})"

    print(f"OK: TTFT reported = {ttft_reported:.2f}s (delay = {DELAY}s)")
    print("OK: order = annonce -> bloc fermé (TTFT) -> premier token")
finally:
    for proc in (delay_backend, proxy):
        try:
            proc.kill()
        except Exception:
            pass
    delay_backend.wait(timeout=5); proxy.wait(timeout=5)
