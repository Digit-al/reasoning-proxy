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
  ``delta.reasoning`` chunks forming an identifiable "CHOIX DU ROUTEUR"
  (router's choice) block in Open WebUI's collapsible thinking box:
  the detected task, the injected effort (and its scale), the origin
  of the decision (sidecar / client / default) and the backend chosen
  (with model substitution). When the creative backend is down and the
  request is retried on the main backend, the block is rebuilt with a
  "Note: …" line explaining the fallback. The block is closed right
  after the first backend byte, with the measured TTFT (time from
  request arrival to first backend byte). Disable with ``EFFORT_NOTIFY=0``.
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
    # Reasoning/agent task effort scale (main backend template).
    "efforts": [e.lower() for e in _env("EFFORTS", "low medium xhigh").split()],
    # Creative task effort scale (creative backend template). May differ
    # from the reasoning scale (e.g. creative LLM uses "high" not "xhigh").
    "creative_efforts": [e.lower() for e in _env("CREATIVE_EFFORTS", "low medium high").split()],
    # Optional dedicated backend for CREATIVE tasks. When empty, creative
    # tasks are handled by the main (reasoning) backend.
    "creative_backend": _env("CREATIVE_BACKEND", "").rstrip("/"),
    "creative_backend_key": _env("CREATIVE_BACKEND_KEY", ""),
    # Model name to use on the creative backend (empty = keep the model
    # requested by the client, assuming it is loaded there too).
    "creative_model": _env("CREATIVE_MODEL", ""),
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


def _route_block(lines: list[str]) -> list[str]:
    """Open the identifiable routing block in the reasoning box.

    The lines are embedded in a SINGLE text with real newlines: Open WebUI
    concatenates ``delta.reasoning`` chunks as-is, so separate chunks per
    line would be rendered glued together on one line.
    """
    return ["\n===== CHOIX DU ROUTEUR =====\n" + "\n".join(lines)]


def _route_close(ttft: float | None = None) -> list[str]:
    """Close the routing block, with the measured TTFT when available."""
    ttft_line = f"TTFT: {ttft:.2f} s\n" if ttft is not None else ""
    return ["\n" + ttft_line + "===== FIN =====\n"]


def _announce_route(model: Any, task: str | None, effort: str | None,
                    origin: str, backend: str, note: str | None = None,
                    scale: str | None = None) -> bytes:
    """Build the reasoning-box announcement of a routing decision.

    The block is left OPEN: it is closed (``TTFT`` + ``===== FIN =====``)
    by the streaming generator right after the first backend byte, so the
    TTFT (arrival of the request -> first byte from the backend) is
    reported to the client.
    """
    if scale is None:
        scale = "échelle créative" if task == "creative" else "échelle raisonnement"
    origin_fr = {
        "sidecar": "sidecar (LLM d'analyse)",
        "default": "EFFORT par défaut (sidecar indisponible)",
        "fallback": "EFFORT par défaut (sidecar indisponible)",
        "client": "spécifié par le client",
        "passthrough": "aucun effort (transfert tel quel)",
    }.get(origin, origin)
    lines = _route_block([
        f"Tâche: {task or 'raisonnement'}",
        f"Effort: {effort or '—'} ({scale})",
        f"Décision: {origin_fr}",
        f"Backend: {backend}",
    ] + ([f"Note: {note}"] if note else []))
    return _reasoning_prefix(lines, model)


if CONFIG["default_effort"] and CONFIG["default_effort"] not in CONFIG["efforts"]:
    CONFIG["efforts"].append(CONFIG["default_effort"])

_DEFAULT_PROMPT = (
    "You are a prompt classifier. You will be given the end of a conversation "
    "(the most recent message is LAST). Classify the user's LAST message on "
    "two dimensions, then answer with a SINGLE JSON object and nothing else.\n\n"
    "1) Task type, exactly one of:\n"
    "- \"creative\": writing, storytelling, brainstorming, marketing copy, "
    "lyrics, role-play, open-ended expression, rewriting for style.\n"
    "- \"reasoning\": analysis, math/logic, coding, debugging, planning, "
    "multi-step or agentic tasks, fact lookup — anything where correctness "
    "matters more than style.\n\n"
    "2) Effort for that task, exactly one value from the scale that matches "
    "the task type:\n"
    "- task \"reasoning\" -> one of: {reasoning_efforts}\n"
    "- task \"creative\"  -> one of: {creative_efforts}\n"
    "Guidance: low = simple request, short direct output; medium = standard "
    "request needing some thought or a full piece of text; {max_effort} = "
    "exceptional complexity (long-form work, deep planning, tricky edge "
    "cases, multi-part structure).\n\n"
    "Answer format: {{\"task\": \"creative|reasoning\", \"effort\": \"<effort>\"}}"
)


def _load_sidecar_prompt() -> str:
    """Load the sidecar system prompt from SIDECAR_PROMPT_FILE, if present.

    Placeholders: {reasoning_efforts}, {creative_efforts}, {max_effort} and
    legacy {efforts} (union of both scales). A missing file falls back to
    the built-in default template.
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
    scale = sorted(set(CONFIG["efforts"] + CONFIG["creative_efforts"]), key=len, reverse=True)
    return (
        template.replace("{reasoning_efforts}", " / ".join(CONFIG["efforts"]))
        .replace("{creative_efforts}", " / ".join(CONFIG["creative_efforts"]))
        .replace("{max_effort}", scale[0])
        .replace("{efforts}", " / ".join(scale))
    )


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


def _parse_json_answer(content: str) -> dict | None:
    """Parse the sidecar JSON answer, tolerating fences/prose around it."""
    s = (content or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    m = re.search(r"\{.*\}", s, re.DOTALL)  # first {...} block
    candidate = m.group(0) if m else s
    try:
        data = json.loads(candidate)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _normalize_effort(value: Any, scale: list[str]) -> str | None:
    """Map a sidecar effort token onto the scale for that task type.

    Handles the usual aliases: `high` == `xhigh` when the scale has no
    `high`, and vice-versa (creative templates often use `high` where
    reasoning templates use `xhigh`).
    """
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in scale:
        return v
    aliases = {"high": "xhigh", "xhigh": "high", "medium": "med"}
    for alt in aliases.get(v, []):
        if alt in scale:
            return alt
    m = re.search(r"\b(" + "|".join(re.escape(e) for e in sorted(scale, key=len, reverse=True)) + r")\b", v)
    return m.group(1) if m else None


def parse_sidecar_answer(content: str) -> tuple[str, str | None] | None:
    """Parse the sidecar answer into (task, effort) or None.

    Accepted formats:
    - JSON: {"task": "creative", "effort": "high"}
    - Legacy single word: `xhigh` (task defaults to "reasoning")
    """
    data = _parse_json_answer(content)
    if data is not None:
        task_raw = data.get("task") or data.get("type") or data.get("type_de_tache")
        task = ""
        if isinstance(task_raw, str):
            task = task_raw.strip().lower()
            if any(w in task for w in ("crea", "créa")):
                task = "creative"
            elif any(w in task for w in ("reason", "réfl", "agent", "logique")):
                task = "reasoning"
            else:
                task = "reasoning" if "reason" in task or "réfl" in task else "reasoning"
        effort = _normalize_effort(
            data.get("effort") or data.get("reasoning_effort"),
            CONFIG["efforts"] + CONFIG["creative_efforts"],
        )
        # task absent (or unknown): only the effort is usable
        if task not in ("creative", "reasoning"):
            task = "reasoning"
        if effort is not None:
            return task, effort
        return None
    # Legacy: single effort word, no task.
    m = re.search(
        r"\b(" + "|".join(
            re.escape(e) for e in sorted(
                CONFIG["efforts"] + CONFIG["creative_efforts"], key=len, reverse=True
            )
        ) + r")\b",
        (content or "").lower(),
    )
    if m:
        return "reasoning", m.group(1)
    return None


def pick_effort_for(task: str, effort: str | None) -> str | None:
    """Validate the effort against the scale of the chosen task type."""
    scale = CONFIG["creative_efforts"] if task == "creative" else CONFIG["efforts"]
    if effort:
        e = _normalize_effort(effort, scale)
        if e:
            return e
    # Best effort in the scale for that task.
    if task == "creative":
        return "medium" if "medium" in scale else (scale[1] if len(scale) > 1 else scale[0])
    return "medium" if "medium" in scale else (scale[1] if len(scale) > 1 else scale[0])


async def ask_sidecar(prompt_text: str, sidecar: httpx.AsyncClient) -> tuple[tuple[str, str | None], float]:
    """Ask the sidecar model to classify the last message as
    ``(task, effort)`` via a JSON answer."""
    url = f"{CONFIG['sidecar_url']}/chat/completions"
    user_content = (
        f"Conversation (most recent message LAST):\n"
        f"{prompt_text}\n\n"
        "Classify the LAST user message and answer with a single "
        'JSON object: {"task": "creative|reasoning", ' + '"effort": "<effort>"}'
    )
    payload = {
        "model": CONFIG["sidecar_model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
    # Log the EXACT context sent to the sidecar and its RAW answer, so a
    # misclassification can be reproduced/diagnosed later.
    log.info("sidecar request context:\n%s", prompt_text)

    t0 = time.monotonic()
    resp = await sidecar.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    msg = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        msg = {}
    log.info("sidecar raw answer (dt=%.2fs): %r", time.monotonic() - t0, msg)
    # Look for the answer in every place a model may put it: `content` is
    # sometimes empty when the backend streams its reasoning elsewhere
    # (`reasoning_content`, `reasoning`, `text`, `raw` …).
    candidates: list[str] = []
    for field in ("content", "reasoning_content", "reasoning", "text", "raw"):
        v = msg.get(field)
        if isinstance(v, str) and v.strip():
            candidates.append(v)
    parsed = None
    for cand in candidates:
        parsed = parse_sidecar_answer(cand)
        if parsed:
            break
    dt = time.monotonic() - t0
    if parsed is None:
        log.warning("sidecar answer not recognized: message=%r", msg)
        raise ValueError(
            f"sidecar returned unrecognized answer: {msg.get('content')!r} "
            f"(full: {msg!r})"
        )
    return parsed, dt


async def decide_route(
    body: dict, sidecar: httpx.AsyncClient
) -> tuple[str | None, str | None, str]:
    """Return (task, effort, source) for a request with no client effort.

    ``task`` is "reasoning" or "creative" (None when no routing info could
    be produced); ``effort`` is validated against the scale of the task.
    """
    conv = build_conversation_text(body)
    if not CONFIG["sidecar_model"] or not conv.strip():
        if CONFIG["default_effort"]:
            return (
                "reasoning",
                pick_effort_for("reasoning", CONFIG["default_effort"]),
                "default",
            )
        return None, None, "passthrough"
    try:
        (task, effort), dt = await ask_sidecar(conv, sidecar)
        effort = pick_effort_for(task, effort)
        log.info("sidecar decided task=%s effort=%s (%.2fs)", task, effort, dt)
        return task, effort, "sidecar"
    except Exception as exc:
        log.warning("sidecar unavailable (%s)", exc)
        if CONFIG["default_effort"]:
            return (
                "reasoning",
                pick_effort_for("reasoning", CONFIG["default_effort"]),
                "fallback",
            )
        return None, None, "passthrough"


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
        follow_redirects=True,
    )
    creative_backend: httpx.AsyncClient | None = None
    if CONFIG["creative_backend"]:
        creative_backend = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
            follow_redirects=True,
        )
    sidecar = httpx.AsyncClient(timeout=CONFIG["sidecar_timeout"])
    app.state.backend = backend
    app.state.creative_backend = creative_backend
    app.state.sidecar = sidecar
    log.info(
        "backend=%s creative_backend=%s sidecar=%s model=%s "
        "efforts=%s creative_efforts=%s default=%r",
        CONFIG["backend"] or "(not set)",
        CONFIG["creative_backend"] or "(not set)",
        CONFIG["sidecar_url"] or "(not set)",
        CONFIG["sidecar_model"] or "(not set)",
        CONFIG["efforts"],
        CONFIG["creative_efforts"],
        CONFIG["default_effort"],
    )
    try:
        yield
    finally:
        await backend.aclose()
        if creative_backend:
            await creative_backend.aclose()
        await sidecar.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/proxy-health")
async def proxy_health() -> dict:
    return {
        "status": "ok",
        "backend": CONFIG["backend"] or None,
        "creative_backend": CONFIG["creative_backend"] or None,
        "creative_model": CONFIG["creative_model"] or None,
        "sidecar": CONFIG["sidecar_model"] or None,
        "efforts": CONFIG["efforts"],
        "creative_efforts": CONFIG["creative_efforts"],
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

    # TTFT reference: when the request reached the proxy.
    arrived_at = time.time()

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

    # Creative tasks may be routed to a dedicated backend; decide before
    # building the outbound request.
    target_client: httpx.AsyncClient = backend
    routed_creative = False
    injected_effort: str | None = None
    original_model = body.get("model") if isinstance(body, dict) else None

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
                reasoning_prefix = _announce_route(
                    body.get("model"),
                    "reasoning",
                    given.strip(),
                    "client",
                    target,
                )
        elif thinking_disabled(body):
            log.info(
                "enable_thinking=false -> forwarding prompt as-is (no sidecar, no injection)"
            )
        else:
            task, effort, source = await decide_route(body, sidecar)
            if effort is not None:
                # Route creative tasks to the dedicated backend when one is
                # configured; otherwise everything stays on the main one.
                use_creative = (
                    task == "creative"
                    and bool(CONFIG["creative_backend"])
                    and request.app.state.creative_backend is not None
                )
                effective_task = "creative" if use_creative else "reasoning"
                effort = pick_effort_for(effective_task, effort)
                if use_creative:
                    target_client = request.app.state.creative_backend
                    routed_creative = True
                    headers = dict(headers)
                    if CONFIG["creative_backend_key"]:
                        headers["Authorization"] = (
                            f"Bearer {CONFIG['creative_backend_key']}"
                        )
                    target = CONFIG["creative_backend"] + request.url.path
                    if request.url.query:
                        target = f"{target}?{request.url.query}"
                    if CONFIG["creative_model"]:
                        body = dict(body)
                        body["model"] = CONFIG["creative_model"]
                    log.info(
                        "routing %s task to creative backend %s",
                        task, CONFIG["creative_backend"],
                    )
                merged = dict(body)
                ctk = merged.get("chat_template_kwargs")
                ctk = dict(ctk) if isinstance(ctk, dict) else {}
                ctk["reasoning_effort"] = effort
                merged["chat_template_kwargs"] = ctk
                forwarded = json.dumps(merged).encode()
                injected_effort = effort
                log.info(
                    "task=%s reasoning_effort=%s injected (source=%s)",
                    task or "reasoning", effort, source,
                )
                if CONFIG["effort_notify"] and body.get("stream") is True:
                    # Show the task the sidecar *detected* (not the effective
                    # one used for routing) and explain any downgrade: a
                    # creative task with no creative backend available is
                    # handled by the main LLM on the reasoning scale.
                    note = None
                    if effective_task != task:
                        note = (
                            "pas de backend créatif disponible -> "
                            "traitée comme reasoning (backend principal)"
                        )
                    reasoning_prefix = _announce_route(
                        body.get("model"),
                        task,
                        effort,
                        source,
                        target,
                        note=note,
                        scale=(
                            "échelle créative"
                            if effective_task == "creative"
                            else "échelle raisonnement"
                        ),
                    )
            else:
                log.info("no effort -> forwarding as-is (source=%s)", source)

    # ---- transparent forwarding (streaming or not) ----
    log.debug("%s %s -> %s", request.method, request.url.path, target)
    client_req = target_client.build_request(
        request.method,
        target,
        headers=headers,
        content=forwarded if raw else None,
    )
    try:
        resp = await target_client.send(client_req, stream=True)
    except httpx.HTTPError as exc:
        # Creative backend unreachable (or hard error before the response
        # started): fall back to the main backend so a downed creative LLM
        # never breaks chat — the prompt still gets a (reasoning-scale)
        # effort and reaches the main LLM.
        if routed_creative:
            log.warning(
                "creative backend %s unavailable (%s) -> retrying on main backend",
                CONFIG["creative_backend"], exc,
            )
            target_client = backend
            headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in SKIP_HEADERS
            }
            if CONFIG["backend_key"] and not any(
                k.lower() == "authorization" for k in headers
            ):
                headers["Authorization"] = f"Bearer {CONFIG['backend_key']}"
            target = CONFIG["backend"] + request.url.path
            if request.url.query:
                target = f"{target}?{request.url.query}"
            fallback_effort = None
            if injected_effort is not None:
                merged = dict(body)
                if original_model is not None:
                    # Restore the client's model: the creative model name
                    # is not loaded on the main backend.
                    merged["model"] = original_model
                else:
                    merged.pop("model", None)
                ctk = merged.get("chat_template_kwargs")
                ctk = dict(ctk) if isinstance(ctk, dict) else {}
                fallback_effort = pick_effort_for("reasoning", injected_effort)
                ctk["reasoning_effort"] = fallback_effort
                merged["chat_template_kwargs"] = ctk
                forwarded = json.dumps(merged).encode()
            else:
                forwarded = raw
            # Re-announce the routing decision: the backend actually used
            # changed (creative backend down -> main backend). The block is
            # rebuilt (it was opened before the first send attempt) so the
            # client clearly sees why the task went to the other LLM.
            if body.get("stream") is True:
                reasoning_prefix = _announce_route(
                    original_model,
                    "creative",
                    fallback_effort,
                    source,
                    target,
                    note=(
                        f"backend créatif {CONFIG['creative_backend']} "
                        "indisponible -> repli sur le backend principal"
                    ),
                    scale="échelle raisonnement (recalé)"
                    if fallback_effort
                    else None,
                )
            client_req = target_client.build_request(
                request.method,
                target,
                headers=headers,
                content=forwarded if raw else None,
            )
            resp = await target_client.send(client_req, stream=True)
        else:
            raise

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
                # When a dedicated creative backend is configured, merge
                # its model list so both LLMs appear in Open WebUI.
                if (
                    isinstance(models, list)
                    and CONFIG["creative_backend"]
                    and target_client is backend
                ):
                    creative_models = await _fetch_creative_models(
                        request.app.state.creative_backend
                    )
                    if creative_models:
                        existing = {
                            (m.get("id") or m.get("model")) for m in models if isinstance(m, dict)
                        }
                        for cm in creative_models:
                            if isinstance(cm, dict) and (
                                cm.get("id") or cm.get("model")
                            ) not in existing:
                                models.append(cm)
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
        block_open = bool(reasoning_prefix)
        try:
            if reasoning_prefix:
                # Announce the routing decision (delta.reasoning) before the
                # first backend chunk reaches Open WebUI. The block is left
                # open and closed below, right after the first backend byte.
                yield reasoning_prefix
            async for chunk in resp.aiter_bytes():
                if block_open:
                    block_open = False
                    # First backend byte: close the routing block with the
                    # measured TTFT (request arrival -> first backend byte).
                    ttft = time.time() - arrived_at
                    yield _reasoning_prefix(_route_close(ttft), None)
                yield chunk
        finally:
            if block_open:
                # Stream ended (or errored) before any backend byte:
                # close the block without a TTFT.
                yield _reasoning_prefix(_route_close(), None)
            await resp.aclose()

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        headers=out_headers,
        background=BackgroundTask(_close_response, resp),
    )


async def _fetch_creative_models(creative: httpx.AsyncClient | None) -> list[dict] | None:
    """Fetch the creative backend's model list (best-effort, merged into
    the main list so both LLMs show up in Open WebUI)."""
    if creative is None:
        return None
    for base in ("/v1/models", "/models"):
        try:
            r = await creative.get(f"{CONFIG['creative_backend']}{base}")
            if r.status_code == 200:
                payload = r.json()
                models = None
                if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    models = payload["data"]
                elif isinstance(payload, list):
                    models = payload
                if models is not None:
                    # Normalize so both mgmt ("model") and v1 ("id") lists
                    # show the model to Open WebUI.
                    for m in models:
                        if isinstance(m, dict):
                            m["id"] = m.get("id") or m.get("model")
                            m["model"] = m.get("model") or m.get("id")
                    return models
        except Exception as exc:
            log.warning("creative backend model list unavailable (%s): %s", base, exc)
            break
    return None


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
