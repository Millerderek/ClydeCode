"""
ClydeCodeBot Web API
Exposes an SSE streaming HTTP endpoint so the OpenClaw web app can talk
to the same SessionManager (and therefore the same session) as Telegram.

Listens on 127.0.0.1:8765 — proxied by Next.js API routes, never public.

Endpoints:
  GET  /health          → {"status":"ok"}
  GET  /status          → {"busy":true,"source":"telegram"} or {"busy":false}
  POST /chat            → SSE stream of {"type":"token","text":"…"} + {"type":"done"}
  POST /stop            → cancel active stream for owner → {"ok":true}
"""

import asyncio
import json
import logging
import os

from aiohttp import web

logger = logging.getLogger("web_api")

# ─── Active stream registry ──────────────────────────────────────────────────
# Maps owner_id → the asyncio.Task running query_streaming for that user.
# Allows /stop to cancel mid-flight (from any interface).
_active_tasks: dict[int, asyncio.Task] = {}


# ─── Token ──────────────────────────────────────────────────────────────────

def _get_token() -> str:
    return os.environ.get("WEB_API_TOKEN", "")


def _check_auth(request: web.Request) -> bool:
    token = _get_token()
    if not token:
        return True  # no token set → open (only safe because it's loopback-only)
    auth = request.headers.get("Authorization", "")
    supplied = auth.removeprefix("Bearer ").strip()
    return supplied == token


# ─── Handlers ───────────────────────────────────────────────────────────────

async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_status(request: web.Request) -> web.Response:
    """Check if the session is currently busy (via Redis busy signal)."""
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)

    sessions = request.app["sessions"]
    owner_id = request.app["owner_id"]
    source = sessions.get_busy(owner_id)
    if source:
        return web.json_response({"busy": True, "source": source})
    return web.json_response({"busy": False, "source": None})


async def handle_stop(request: web.Request) -> web.Response:
    """Cancel the active streaming task for the owner, if any."""
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)

    owner_id = request.app["owner_id"]
    task = _active_tasks.get(owner_id)
    if task and not task.done():
        task.cancel()
        logger.info("Web API: stream cancelled for owner %d", owner_id)
        return web.json_response({"ok": True, "cancelled": True})

    return web.json_response({"ok": True, "cancelled": False})


async def handle_chat(request: web.Request) -> web.StreamResponse:
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    message = body.get("message", "").strip()
    if not message:
        return web.json_response({"error": "message required"}, status=400)

    sessions   = request.app["sessions"]
    owner_id   = request.app["owner_id"]

    # SSE response
    resp = web.StreamResponse(headers={
        "Content-Type":  "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)

    client_gone = False

    async def send(chunk: dict):
        nonlocal client_gone
        if client_gone:
            return
        try:
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        except Exception:
            client_gone = True  # client disconnected — stop writing, don't raise

    # Delta-based streaming: on_chunk receives typed deltas directly,
    # no phase tracking or accumulated-text deduplication needed.
    async def on_chunk(chunk_type: str, delta: str):
        if not delta:
            return
        if chunk_type == "thinking":
            await send({"type": "thinking", "text": delta})
        elif chunk_type == "text":
            await send({"type": "token", "text": delta})
        elif chunk_type == "tool":
            # Suppress tool status from web UI — thinking spinner covers this
            pass

    # Wrap query_streaming in a Task so /stop can cancel it
    async def run_query():
        await sessions.query_streaming(owner_id, message, on_chunk=on_chunk, source="web")

    task = asyncio.create_task(run_query())
    _active_tasks[owner_id] = task

    try:
        await task
        await send({"type": "done", "messageId": ""})
    except asyncio.CancelledError:
        logger.info("Web API: stream for owner %d was cancelled", owner_id)
        await send({"type": "stopped"})
    except Exception as e:
        logger.error("Web API stream error: %s", e)
        await send({"type": "error", "error": str(e)})
    finally:
        _active_tasks.pop(owner_id, None)

    try:
        await resp.write_eof()
    except Exception:
        pass  # client already closed — safe to ignore
    return resp


# ─── App factory ────────────────────────────────────────────────────────────

def create_app(sessions, owner_id: int) -> web.Application:
    app = web.Application()
    app["sessions"] = sessions
    app["owner_id"] = owner_id
    app.router.add_get("/health",  handle_health)
    app.router.add_get("/status",  handle_status)
    app.router.add_post("/chat",   handle_chat)
    app.router.add_post("/stop",   handle_stop)
    return app


async def start_web_api(sessions, owner_id: int, port: int = 8765):
    """Start the aiohttp server in the current event loop (non-blocking)."""
    app = create_app(sessions, owner_id)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    logger.info("Web API listening on http://127.0.0.1:%d", port)
    return runner  # caller holds reference to prevent GC
