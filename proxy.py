#!/usr/bin/env python3
"""
Reasoning-effort proxy:  Open WebUI  ->  proxy  ->  llama.cpp
=================

A transparent, OpenAI-compatible passthrough proxy that decides how much
reasoning effort a prompt needs, and tells llama.cpp through
``chat_template_kwargs.reasoning_effort``.

Behaviour
---------
* ``POST */chat/completions`` with NO effort provided anywhere:
    the proxy asks a small "sidecar" LLM (any OpenAI-compatible endpoint)
    which effort (low / medium / high) the current message deserves, then
    injects ``chat_template_kwargs.reasoning_effort`` into the request.
* ``POST */chat/completions`` with ``reasoning_effort`` already present
    (either in ``chat_template_kwargs`` or top-level — this is what
    Open WebUI sends when the user fills the "Advanced options" field):
    the body is forwarded byte-for-byte, the sidecar is NOT consulted.
* ``POST */chat/completions`` with
  ``chat_template_kwargs.enable_thinking = false``:
    thinking is explicitly disabled, so reasoning effort is meaningless —
    the request (prompt included) is forwarded as-is to the main LLM,
    the sidecar is NOT consulted and no ``reasoning_effort`` is injected.
* Sidecar failure / timeout: falls back to ``DEFAULT_REASONING_EFFORT``
    (if set), otherwise the request is forwarded as-is.
* Streaming responses (``stream: true``) are prefixed with
  ``delta.reasoning`` chunks announcing the effort —
  "D\u00e9termination de l'effort de raisonnement\u2026" then
  "Effort: {effort}" — so Open WebUI shows it in its collapsible
  thinking box before the answer starts. When the client supplies its
  own effort, only the echo "Effort: {effort}" is sent. Disable with
  ``EFFORT_NOTIFY=0``.
* ``GET /v1/models`` and ``GET /models`` (model lists): each model is
  reported with ``"loaded": true`` and
  ``"status": {"value": "loaded"}`` when the backend does not provide
  them. This makes Open WebUI display the green "loaded" dot for the
  proxy's models exactly like a direct llama.cpp connection — with any
  provider setting ("Défaut" included). Other ``/models/*`` management
  routes are proxied untouched.
* Everything else (``/v1/models``, audio, tools, streaming, headers,
  status codes…) is proxied completely untouched — including SSE streams.

Configuration (environment variables)
-------------------------------------
PROXY_HOST                  bind address          (default 0.0.0.0)
PROXY_PORT                  bind port             (default 8080)
LLAMA_BACKEND               llama.cpp base URL, no version suffix,
                            e.g. http://127.0.0.1:8081 (required)
LLAMA_BACKEND_KEY           optional API key for the backend
SIDECAR_BASE_URL            sidecar OpenAI-compatible base URL
                            e.g. http://127.0.0.1:8082/v1 (required)
SIDECAR_API_KEY             optional API key for the sidecar
SIDECAR_MODEL               sidecar model name   (required)
SIDECAR_TIMEOUT             sidecar HTTP timeout  (default 10 s)
EFFORTS                     allowed values        (default "low medium high")
EFFORTS                     allowed effort values (default
                            "low medium xhigh"). Drives BOTH the sidecar
                            prompt (the {efforts} placeholder) and the
                            parser that reads the sidecar answer.
SIDECAR_PROMPT_FILE         path to the sidecar system prompt template
                            (default: sidecar_prompt.txt next to
                            proxy.py). The placeholder {efforts} is
                            replaced with the EFFORTS list at startup.
                            Hardcoding the words in the file is also
                            fine — EFFORTS only governs parsing then.
DEFAULT_REASONING_EFFORT    fallback effort when the sidecar fails /
                            is not configured; empty = forward as-is
CONTEXT_TURNS               how many trailing turns to send to the sidecar
                            (default 2)
MAX_PROMPT_CHARS            char budget sent to the sidecar (default 6000)
EFFORT_NOTIFY               announce the effort in the stream
                            (default on)
LOG_LEVEL                   default INFO
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [reasoning-proxy] %(message)s",
)
log = logging.getLogger("reasoning-proxy")


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


CONFIG: dict[str, Any] = {
    "host": _env("PROXY_HOST", "0.0.0.0"),
    "port": int(_env("PROXY_PORT", "8080")),
    "backend": _env("LLAMA_BACKEND", "").rstrip("/"),
    "backend_key": _env("LLAMA_BACKEND_KEY", ""),
    "sidecar_url": _env("SIDECAR_BASE_URL", "").rstrip("/"),
    "sidecar_key": _env("SIDECAR_API_KEY", ""),
    "sidecar_model": _env("SIDECAR_MODEL", ""),
    "sidecar_timeout": float(_env("SIDECAR_TIMEOUT", "10")),
    "efforts": [e.lower() for e in _env("EFFORTS", "low medium xhigh").split()],
    "prompt_file": _env(
        "SIDECAR_PROMPT_FILE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sidecar_prompt.txt"),
    ),
    "default_effort": _env("DEFAULT_REASONING_EFFORT", "").strip().lower(),
    "effort_notify": _env("EFFORT_NOTIFY", "1") not in ("0", "false", "no", "off"),
    "context_turns": max(1, int(_env("CONTEXT_TURNS", "2"))),
    "max_prompt_chars": int(_env("MAX_PROMPT_CHARS", "6000")),
}


def _reasoning_prefix(announce_texts: list[str], model: Any) -> bytes:
    """Build SSE chunks announcing the effort via ``delta.reasoning``.

    Open WebUI renders ``delta.reasoning`` in its collapsible "thinking"
    box, so the effort shows up there without polluting the message
    content or the stored conversation history.
    """
    cid = f"rproxy-{time.time_ns()}"
    out = bytearray()
    for i, text in enumerate(announce_texts):
        delta = {"reasoning": text}
        if i == 0:
            delta = {"role": "assistant", **delta}
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model if isinstance(model, str) and model else "reasoning-proxy",
            "choices": [{"index": 0, "delta": delta}],
        }
        out += f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
    return bytes(out)
if CONFIG["default_effort"] and CONFIG["default_effort"] not in CONFIG["efforts"]:
    CONFIG["efforts"].append(CONFIG["default_effort"])

_DEFAULT_PROMPT = (
    "You are a prompt complexity classifier. You will be given the end of a "
    "conversation (the most recent message is LAST). Decide how much "
    "reasoning effort an LLM needs to answer the user's LAST message well.\n"
    "Allowed values: {efforts}\n"
    "- low: simple facts, greetings, short direct answers, lookups, trivial "
    "code, rewriting/summarizing short text.\n"
    "- medium: standard questions needing some thought, routine multi-step "
    "tasks, short code with a clear goal.\n"
    "- xhigh: complex math/logic, proofs, debugging, architecture/planning, "
    "long-form analysis, tricky edge cases, multi-part reasoning.\n"
    "Answer with exactly one word: {efforts}. No explanation, just the word."
)


def _load_sidecar_prompt() -> str:
    """Load the sidecar system prompt from SIDECAR_PROMPT_FILE, if present.

    The placeholder {efforts} is replaced with the configured EFFORTS list.
    A missing file falls back to the built-in default template.
    """
    path = CONFIG["prompt_file"]
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            template = fh.read().strip()
        log.info("sidecar prompt loaded from %s", path)
    else:
        if path:
            log.warning("sidecar prompt file %s not found -> built-in default", path)
        template = _DEFAULT_PROMPT
    return template.replace("{efforts}", " / ".join(CONFIG["efforts"]))


SYSTEM_PROMPT = _load_sidecar_prompt()

# request headers we must not forward verbatim (httpx sets its own)
SKIP_HEADERS = {
    "host", "content-length", "connection", "keep-alive",
    "proxy-authorization", "proxy-connection", "te", "trailer",
}
# response headers we must not copy (recomputed by the ASGI server)
SKIP_RESP_HEADERS = {"transfer-encoding", "content-length", "connection"}


def client_effort(body: dict) -> Any | None:
    """Return the client-provided reasoning_effort, if any."""
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and ctk.get("reasoning_effort") is not None:
        return ctk["reasoning_effort"]
    if body.get("reasoning_effort") is not None:
        return body["reasoning_effort"]
    return None


def thinking_disabled(body: dict) -> bool:
    """True when the client explicitly turned thinking off via
    ``chat_template_kwargs.enable_thinking = false`` — in that case the
    prompt must reach the main LLM untouched (no effort injection)."""
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict):
        v = ctk.get("enable_thinking")
        if v is False or (isinstance(v, str) and v.strip().lower() == "false"):
            return True
    return False


def build_conversation_text(body: dict) -> str:
    """Format the trailing turns of the conversation for the sidecar."""
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return ""
    max_turns = CONFIG["context_turns"]
    max_chars = CONFIG["max_prompt_chars"]
    last_budget = int(max_chars * 0.8)

    texts: list[tuple[str, str]] = []
    for m in reversed(msgs):  # newest first
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant", "system"):
            continue
        content = m.get("content")
        if isinstance(content, list):  # OpenAI "parts" style
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type", "text") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            continue
        texts.append((role, content))
        if len(texts) >= max_turns:
            break

    parts: list[str] = []
    n = len(texts)
    for i, (role, content) in enumerate(reversed(texts)):  # oldest -> newest
        is_last = i == n - 1
        cap = last_budget if is_last else max(500, (max_chars - last_budget) // max(1, n - 1))
        if len(content) > cap:
            content = (
                f"…[truncated]…{content[-cap:]}" if is_last else f"{content[:cap]}…[truncated]…"
            )
        parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


def parse_effort(content: str) -> str | None:
    """Extract the first recognized effort value from a sidecar answer."""
    m = re.search(
        r"\b(" + "|".join(re.escape(e) for e in sorted(CONFIG["efforts"], key=len, reverse=True)) + r")\b",
        (content or "").lower(),
    )
    return m.group(1) if m else None


async def ask_sidecar(prompt_text: str, sidecar: httpx.AsyncClient) -> tuple[str, float]:
    """Ask the sidecar model which reasoning effort to use."""
    url = f"{CONFIG['sidecar_url']}/chat/completions"
    payload = {
        "model": CONFIG["sidecar_model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation (most recent message LAST):\n"
                    f"{prompt_text}\n\n"
                    f"Classify the LAST user message. Answer with exactly one word: "
                    f"{' / '.join(CONFIG['efforts'])}."
                ),
            },
        ],
        "max_tokens": 64,
        "temperature": 0,
        "stream": False,
        # The sidecar is a quick classifier: never let it *think*. A
        # reasoning model started with `--reasoning on` would otherwise
        # burn its tokens on a thinking trace and return an empty
        # `content`. Disabling thinking makes it answer the single word
        # directly, whether the backend is `--reasoning on` or `off`.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {}
    if CONFIG["sidecar_key"]:
        headers["Authorization"] = f"Bearer {CONFIG['sidecar_key']}"
    t0 = time.monotonic()
    resp = await sidecar.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    msg = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        msg = {}
    # Look for the answer in every place a model may put it: `content` is
    # sometimes empty when the backend streams its reasoning elsewhere
    # (`reasoning_content`, `reasoning`, `text`, `raw` …).
    candidates: list[str] = []
    for field in ("content", "reasoning_content", "reasoning", "text", "raw"):
        v = msg.get(field)
        if isinstance(v, str) and v.strip():
            candidates.append(v)
    effort = None
    for cand in candidates:
        effort = parse_effort(cand)
        if effort:
            break
    dt = time.monotonic() - t0
    if effort is None:
        log.warning("sidecar answer not recognized: message=%r", msg)
        raise ValueError(
            f"sidecar returned unrecognized effort: {msg.get('content')!r} "
            f"(full: {msg!r})"
        )
    return effort, dt


async def decide_effort(body: dict, sidecar: httpx.AsyncClient) -> tuple[str | None, str]:
    """Return (effort_or_None, source) for a request with no client effort."""
    conv = build_conversation_text(body)
    if not CONFIG["sidecar_model"] or not conv.strip():
        if CONFIG["default_effort"]:
            return CONFIG["default_effort"], "default"
        return None, "passthrough"
    try:
        effort, dt = await ask_sidecar(conv, sidecar)
        log.info("sidecar decided effort=%s (%.2fs)", effort, dt)
        return effort, "sidecar"
    except Exception as exc:
        log.warning("sidecar unavailable (%s)", exc)
        if CONFIG["default_effort"]:
            return CONFIG["default_effort"], "fallback"
        return None, "passthrough"


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
        follow_redirects=True,
    )
    sidecar = httpx.AsyncClient(timeout=CONFIG["sidecar_timeout"])
    app.state.backend = backend
    app.state.sidecar = sidecar
    log.info(
        "backend=%s sidecar=%s model=%s efforts=%s default=%r",
        CONFIG["backend"] or "(not set)",
        CONFIG["sidecar_url"] or "(not set)",
        CONFIG["sidecar_model"] or "(not set)",
        CONFIG["efforts"],
        CONFIG["default_effort"],
    )
    try:
        yield
    finally:
        await backend.aclose()
        await sidecar.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/proxy-health")
async def proxy_health() -> dict:
    return {
        "status": "ok",
        "backend": CONFIG["backend"] or None,
        "sidecar": CONFIG["sidecar_model"] or None,
        "efforts": CONFIG["efforts"],
        "default_effort": CONFIG["default_effort"] or None,
    }


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    include_in_schema=False,
)
async def proxy(request: Request, path: str):
    backend: httpx.AsyncClient = request.app.state.backend
    sidecar: httpx.AsyncClient = request.app.state.sidecar

    if not CONFIG["backend"]:
        return JSONResponse(
            {"error": "LLAMA_BACKEND is not configured for the proxy"},
            status_code=503,
        )

    raw = await request.body()
    target = CONFIG["backend"] + request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"

    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in SKIP_HEADERS
    }
    if CONFIG["backend_key"] and not any(
        k.lower() == "authorization" for k in headers
    ):
        headers["Authorization"] = f"Bearer {CONFIG['backend_key']}"

    # ---- possible interception: chat completions without an effort ----
    forwarded = raw
    reasoning_prefix = b""  # SSE chunks announced before the backend stream
    is_chat = (
        request.method == "POST"
        and request.url.path.rstrip("/").endswith("/chat/completions")
        and request.headers.get("content-type", "").lower().startswith("application/json")
    )
    body = None
    if is_chat and raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            body = parsed

    if body is not None:
        given = client_effort(body)
        if given is not None:
            log.info("client supplied reasoning_effort=%r -> forwarding as-is", given)
            if (
                CONFIG["effort_notify"]
                and body.get("stream") is True
                and isinstance(given, str)
                and given.strip()
            ):
                reasoning_prefix = _reasoning_prefix(
                    [f"Effort: {given.strip()}"], body.get("model")
                )
        elif thinking_disabled(body):
            log.info(
                "enable_thinking=false -> forwarding prompt as-is (no sidecar, no injection)"
            )
        else:
            effort, source = await decide_effort(body, sidecar)
            if effort is not None:
                merged = dict(body)
                ctk = merged.get("chat_template_kwargs")
                ctk = dict(ctk) if isinstance(ctk, dict) else {}
                ctk["reasoning_effort"] = effort
                merged["chat_template_kwargs"] = ctk
                forwarded = json.dumps(merged).encode()
                log.info("reasoning_effort=%s injected (source=%s)", effort, source)
                if CONFIG["effort_notify"] and body.get("stream") is True:
                    reasoning_prefix = _reasoning_prefix(
                        [
                            "D\u00e9termination de l\u2019effort de raisonnement\u2026\n",
                            f"Effort: {effort}",
                        ],
                        body.get("model"),
                    )
            else:
                log.info("no effort -> forwarding as-is (source=%s)", source)

    # ---- transparent forwarding (streaming or not) ----
    log.debug("%s %s -> %s", request.method, request.url.path, target)
    client_req = backend.build_request(
        request.method,
        target,
        headers=headers,
        content=forwarded if raw else None,
    )
    resp = await backend.send(client_req, stream=True)

    # ---- model list endpoints: report models as loaded to Open WebUI ----
    # OWUI shows the green "loaded" dot when a model carries
    # `loaded: true` (any provider, the field is passed through as-is)
    # or `status.value == "loaded"` (Provider = llama.cpp, read from
    # the management list). Models reached through the proxy are always
    # available, so report them as loaded (unless the backend already
    # says otherwise).
    p = request.url.path
    if p in ("/models", "/v1/models") or p.startswith("/models/"):
        ctype = resp.headers.get("content-type", "").lower()
        if "json" in ctype:
            body_bytes = await resp.aread()
            await resp.aclose()
            data: Any = None
            try:
                data = json.loads(body_bytes)
            except Exception:
                data = None
            if data is not None:
                models = data.get("data") if isinstance(data, dict) else data
                if isinstance(models, list):
                    for m in models:
                        if not isinstance(m, dict):
                            continue
                        if "status" not in m:
                            m["status"] = {"value": "loaded"}
                        if "loaded" not in m:
                            status = m.get("status")
                            value = status.get("value") if isinstance(status, dict) else None
                            m["loaded"] = (
                                value in ("loaded", "sleeping")
                                if value is not None
                                else True
                            )
                    log.info(
                        "marked %d model(s) as loaded for Open WebUI (%s)",
                        len(models),
                        p,
                    )
                return JSONResponse(data, status_code=resp.status_code)
            # Unparseable JSON payload: return the raw bytes unchanged.
            return StreamingResponse(
                iter([body_bytes]),
                status_code=resp.status_code,
                headers={"content-type": "application/json"},
            )

    out_headers: dict[str, str] = {}
    for k, v in resp.headers.items():
        lk = k.lower()
        if lk in SKIP_RESP_HEADERS:
            continue
        if lk in out_headers:  # e.g. multiple set-cookie
            out_headers[lk] += ", " + v
        else:
            out_headers[lk] = v

    async def gen():
        try:
            if reasoning_prefix:
                # Announce the effort (delta.reasoning) before the first
                # backend chunk reaches Open WebUI.
                yield reasoning_prefix
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        headers=out_headers,
        background=BackgroundTask(_close_response, resp),
    )


async def _close_response(resp: httpx.Response) -> None:
    await resp.aclose()


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=CONFIG["host"],
        port=CONFIG["port"],
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
