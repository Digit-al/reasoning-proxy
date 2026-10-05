#!/usr/bin/env python3
"""End-to-end tests for the reasoning-effort proxy (no pytest needed)."""
import json
import os
import signal
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PROXY_PORT, BACKEND_PORT, SIDECAR_PORT, CREATIVE_PORT = 18080, 18081, 18082, 18083
BASE = f"http://127.0.0.1:{PROXY_PORT}"

procs: list[subprocess.Popen] = []
failures: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if not cond else ""))
    if not cond:
        failures.append(name)


def wait_port(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=1.0)
        except httpx.ConnectError:
            time.sleep(0.3)
            continue
        except Exception:
            return True  # something is listening
        else:
            return True
    return False


def start(name: str, script: str, env: dict, port: int):
    p = subprocess.Popen(
        [sys.executable, script],
        env={**os.environ, **env},
        cwd=HERE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    procs.append(p)
    assert wait_port(port), f"{name} did not come up on :{port}"
    print(f"  started {name} on :{port}")


def kill_all():
    for p in procs:
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass
    for p in procs:
        p.wait(timeout=5)


def sse_reasoning(text: str) -> list[str]:
    """Collect every delta.reasoning value from an SSE stream body."""
    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except Exception:
            continue
        for ch in data.get("choices") or []:
            r = (ch.get("delta") or {}).get("reasoning")
            if isinstance(r, str):
                out.append(r)
    return out


def main() -> int:
    print("== starting mocks + proxy ==")
    start("sidecar", "mock_sidecar.py", {"PORT": str(SIDECAR_PORT)}, SIDECAR_PORT)
    start(
        "backend",
        "mock_backend.py",
        {"PORT": str(BACKEND_PORT), "MODEL_ID": "mock-model"},
        BACKEND_PORT,
    )
    start(
        "creative-backend",
        "mock_backend.py",
        {"PORT": str(CREATIVE_PORT), "MODEL_ID": "creative-llm"},
        CREATIVE_PORT,
    )
    start(
        "proxy",
        os.path.join(ROOT, "proxy.py"),
        {
            "PROXY_PORT": str(PROXY_PORT),
            "LLAMA_BACKEND": f"http://127.0.0.1:{BACKEND_PORT}",
            "SIDECAR_BASE_URL": f"http://127.0.0.1:{SIDECAR_PORT}/v1",
            "SIDECAR_MODEL": "sidecar-mini",
            "SIDECAR_TIMEOUT": "5",
            "EFFORTS": "low medium xhigh",
            "CREATIVE_EFFORTS": "low medium high",
            "CREATIVE_BACKEND": f"http://127.0.0.1:{CREATIVE_PORT}",
            "CREATIVE_MODEL": "creative-llm",
            "DEFAULT_REASONING_EFFORT": "medium",
        },
        PROXY_PORT,
    )
    client = httpx.Client(timeout=30)

    # ---- T1: no effort, streaming -> sidecar decides, body injected ----
    print("\n== T1: no effort provided (streaming) -> sidecar decides ==")
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Explain quantum entanglement step by step"}], "stream": True},
    )
    text = r.text
    check("T1 status 200", r.status_code == 200, str(r.status_code))
    check("T1 SSE contains [DONE]", "data: [DONE]" in text)
    check("T1 stream contains model echo", "hello" in text)
    # effort notification: announced in delta.reasoning before the backend chunks
    reasoning = "".join(sse_reasoning(text))
    check(
        "T1 effort announced in reasoning",
        "D\u00e9termination" in reasoning and "xhigh" in reasoning,
        reasoning[:200],
    )

    sidecar_calls = client.get(f"http://127.0.0.1:{SIDECAR_PORT}/calls").json()
    check("T1 sidecar consulted once", sidecar_calls["count"] == 1, str(sidecar_calls["count"]))
    check(
        "T1 sidecar saw the prompt",
        "quantum entanglement" in json.dumps(sidecar_calls["last"]),
    )
    received = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    last = received[-1]
    check(
        "T1 effort injected in chat_template_kwargs",
        (last.get("chat_template_kwargs") or {}).get("reasoning_effort") == "xhigh",
        json.dumps(last.get("chat_template_kwargs")),
    )
    # the sidecar prompt (loaded from sidecar_prompt.txt) must have been
    # filled with the configured efforts and sent as system message
    sidecar_sys = (sidecar_calls["last"].get("messages") or [{}])[0].get("content", "")
    check("T1 sidecar prompt from file with efforts", "xhigh" in sidecar_sys and "{efforts}" not in sidecar_sys, sidecar_sys[:120])
    check("T1 stream flag preserved", last.get("stream") is True)
    # the sidecar request must disable thinking so a reasoning model
    # answers directly whether it runs with --reasoning on or off
    sidecar_ctk = sidecar_calls["last"].get("chat_template_kwargs") or {}
    check(
        "T1 sidecar gets enable_thinking=false",
        sidecar_ctk.get("enable_thinking") is False,
        json.dumps(sidecar_calls["last"].get("chat_template_kwargs")),
    )

    # ---- T2: client sends effort via chat_template_kwargs -> as-is ----
    print("\n== T2: client provides reasoning_effort in chat_template_kwargs -> as-is ==")
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "chat_template_kwargs": {"reasoning_effort": "low", "other_kwarg": 42},
        },
    )
    check("T2 status 200", r.status_code == 200)
    check("T2 response passthrough", r.json()["choices"][0]["message"]["content"].startswith("answer"))
    sidecar_calls = client.get(f"http://127.0.0.1:{SIDECAR_PORT}/calls").json()
    check("T2 sidecar NOT consulted", sidecar_calls["count"] == 1, str(sidecar_calls["count"]))
    received = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    last = received[-1]
    check("T2 effort untouched", last["chat_template_kwargs"]["reasoning_effort"] == "low")
    check("T2 other kwargs preserved", last["chat_template_kwargs"]["other_kwarg"] == 42)

    # ---- T3: no effort, non-streaming -> injection + JSON passthrough ----
    print("\n== T3: no effort provided (non-streaming) ==")
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "2+2?"}], "stream": False},
    )
    check("T3 status 200", r.status_code == 200)
    body = r.json()
    check("T3 response JSON intact", body.get("object") == "chat.completion")
    received = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    # mock sidecar: "2+2?" has no hard/creative keyword -> medium
    check("T3 effort injected", received[-1].get("chat_template_kwargs", {}).get("reasoning_effort") == "medium")

    # ---- T4: tools are preserved ----
    print("\n== T4: tools passthrough ==")
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "stream": False,
            "tools": tools,
            "tool_choice": "auto",
        },
    )
    check("T4 status 200", r.status_code == 200)
    received = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    check("T4 tools preserved", received[-1].get("tools") == tools)
    check("T4 tool_choice preserved", received[-1].get("tool_choice") == "auto")

    # ---- T5: /v1/models passthrough ----
    print("\n== T5: GET /v1/models passthrough ==")
    r = client.get(f"{BASE}/v1/models")
    check("T5 status 200", r.status_code == 200, str(r.status_code))
    check("T5 models list", any(m["id"] == "mock-model" for m in r.json()["data"]))

    # ---- T6: top-level client effort -> as-is (no sidecar) ----
    print("\n== T6: top-level reasoning_effort -> as-is ==")
    before = client.get(f"http://127.0.0.1:{SIDECAR_PORT}/calls").json()["count"]
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False, "reasoning_effort": "high"},
    )
    check("T6 status 200", r.status_code == 200)
    after = client.get(f"http://127.0.0.1:{SIDECAR_PORT}/calls").json()["count"]
    check("T6 sidecar NOT consulted", before == after, f"{before} -> {after}")
    received = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    check("T6 effort forwarded", received[-1].get("reasoning_effort") == "high")
    # client-provided effort is echoed in the reasoning box
    r6 = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True, "reasoning_effort": "high"},
    )
    check("T6 echo status 200", r6.status_code == 200)
    check(
        "T6 client effort echoed in reasoning",
        "Effort: high" in "".join(sse_reasoning(r6.text)),
        "".join(sse_reasoning(r6.text))[:200],
    )

    # ---- T7: enable_thinking=false -> as-is (no sidecar, no injection) ----
    print("\n== T7: enable_thinking=false -> prompt forwarded as-is ==")
    before = client.get(f"http://127.0.0.1:{SIDECAR_PORT}/calls").json()["count"]
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "a hard math proof question"}],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    check("T7 status 200", r.status_code == 200, str(r.status_code))
    after = client.get(f"http://127.0.0.1:{SIDECAR_PORT}/calls").json()["count"]
    check("T7 sidecar NOT consulted", before == after, f"{before} -> {after}")
    received = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    last = received[-1]
    check(
        "T7 no reasoning_effort injected",
        "reasoning_effort" not in (last.get("chat_template_kwargs") or {}),
        json.dumps(last.get("chat_template_kwargs")),
    )
    check(
        "T7 enable_thinking preserved",
        (last.get("chat_template_kwargs") or {}).get("enable_thinking") is False,
        json.dumps(last.get("chat_template_kwargs")),
    )
    check("T7 prompt untouched", last["messages"][0]["content"] == "a hard math proof question")
    # enable_thinking=false: no effort notification at all
    r7s = client.post(
        f"{BASE}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "a hard math proof question"}],
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    check("T7s stream 200", r7s.status_code == 200)
    check(
        "T7s no reasoning announcement (thinking disabled)",
        sse_reasoning(r7s.text) == [],
        str(sse_reasoning(r7s.text)),
    )

    # ---- T8: GET /models -> models reported as loaded (OWUI green dot) ----
    print("\n== T8: GET /models management list -> status loaded injected ==")
    r = client.get(f"{BASE}/models")
    check("T8 status 200", r.status_code == 200, str(r.status_code))
    mgmt = {m["model"]: m for m in r.json()["data"]}
    check(
        "T8 model without status gets status.loaded",
        (mgmt.get("mock-model") or {}).get("status") == {"value": "loaded"},
        json.dumps(mgmt.get("mock-model")),
    )
    check(
        "T8 explicit status not overwritten",
        (mgmt.get("mock-model-2") or {}).get("status", {}).get("value") == "unloaded",
        json.dumps(mgmt.get("mock-model-2")),
    )
    # /v1/models is enriched too -> green dot even with provider "default"
    r = client.get(f"{BASE}/v1/models")
    v1m = r.json()["data"][0]
    check("T8 /v1/models marked loaded", v1m.get("loaded") is True, json.dumps(v1m))
    check("T8 /v1/models status loaded", (v1m.get("status") or {}).get("value") == "loaded")
    check(
        "T8 unloaded model -> loaded false",
        (mgmt.get("mock-model-2") or {}).get("loaded") is False,
        json.dumps(mgmt.get("mock-model-2")),
    )

    # ---- T11: creative task -> routed to creative backend ----
    print("\n== T11: creative task -> creative backend + creative effort ==")
    n_backend = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    n_creative = client.get(f"http://127.0.0.1:{CREATIVE_PORT}/received").json()["bodies"]
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Write a creative poem about the sea"}], "stream": False},
    )
    check("T11 status 200", r.status_code == 200, str(r.status_code))
    n_backend2 = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    n_creative2 = client.get(f"http://127.0.0.1:{CREATIVE_PORT}/received").json()["bodies"]
    check("T11 main backend untouched", len(n_backend2) == len(n_backend))
    check("T11 creative backend received the request", len(n_creative2) == len(n_creative) + 1, f"{len(n_creative)} -> {len(n_creative2)}")
    last_creative = n_creative2[-1] if n_creative2 else {}
    check(
        "T11 creative model name substituted",
        last_creative.get("model") == "creative-llm",
        json.dumps(last_creative.get("model")),
    )
    check(
        "T11 creative effort injected",
        (last_creative.get("chat_template_kwargs") or {}).get("reasoning_effort") == "low",
        json.dumps(last_creative.get("chat_template_kwargs")),
    )

    # ---- T12: creative effort scale mapping (high -> creative, xhigh -> reasoning) ----
    print("\n== T12: effort scales per task type ==")
    # creative task with an effort that only exists on the reasoning scale
    # must be mapped onto the creative scale (xhigh -> high).
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Write an epic creative saga about dragons"}], "stream": False},
    )
    check("T12 status 200", r.status_code == 200, str(r.status_code))
    n_creative3 = client.get(f"http://127.0.0.1:{CREATIVE_PORT}/received").json()["bodies"]
    check(
        "T12 creative effort on creative scale",
        (n_creative3[-1].get("chat_template_kwargs") or {}).get("reasoning_effort") in ("low", "medium", "high"),
        json.dumps(n_creative3[-1].get("chat_template_kwargs")),
    )
    # reasoning task with a creative-scale effort maps onto the reasoning scale.
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Explain quantum entanglement"}], "stream": False},
    )
    check("T13 status 200", r.status_code == 200, str(r.status_code))
    n_backend3 = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    check(
        "T13 reasoning effort on reasoning scale",
        (n_backend3[-1].get("chat_template_kwargs") or {}).get("reasoning_effort") in ("low", "medium", "xhigh"),
        json.dumps(n_backend3[-1].get("chat_template_kwargs")),
    )

    # ---- T14: creative backend down -> transparent fallback to main ----
    print("\n== T14: creative backend dead -> fallback to main backend ==")
    creative_proc = procs[2]  # started third
    creative_proc.send_signal(signal.SIGTERM)
    creative_proc.wait(timeout=5)
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Write a creative poem about the sea"}], "stream": False},
    )
    check("T14 status 200 (main backend rescued the task)", r.status_code == 200, str(r.status_code))
    n_backend4 = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    check(
        "T14 main backend received the creative task",
        n_backend4[-1]["messages"][0]["content"] == "Write a creative poem about the sea",
        n_backend4[-1]["messages"][0]["content"][:80],
    )
    check(
        "T14 effort mapped to reasoning scale",
        (n_backend4[-1].get("chat_template_kwargs") or {}).get("reasoning_effort") in ("low", "medium", "xhigh"),
        json.dumps(n_backend4[-1].get("chat_template_kwargs")),
    )

    # ---- T9: sidecar dead -> fallback DEFAULT_REASONING_EFFORT=medium ----
    print("\n== T9: sidecar dead -> fallback effort ==")
    sidecar_proc = procs[0]  # started first
    sidecar_proc.send_signal(signal.SIGTERM)
    sidecar_proc.wait(timeout=5)
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "a complex multi-step planning question"}], "stream": False},
    )
    check("T9 status 200 (proxy still works)", r.status_code == 200, str(r.status_code))
    received = client.get(f"http://127.0.0.1:{BACKEND_PORT}/received").json()["bodies"]
    check(
        "T9 fallback effort applied",
        received[-1].get("chat_template_kwargs", {}).get("reasoning_effort") == "medium",
        json.dumps(received[-1].get("chat_template_kwargs")),
    )

    # ---- T10: proxy health ----
    print("\n== T10: /proxy-health ==")
    r = client.get(f"{BASE}/proxy-health")
    check("T10 health ok", r.json().get("status") == "ok")
    check("T10 health reports creative backend", (r.json().get("creative_backend") or "").endswith(str(CREATIVE_PORT)), r.json().get("creative_backend"))

    kill_all()
    print("\n== RESULT ==")
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
