#!/usr/bin/env python3
"""
Claude Code Bridge — OpenAI-compatible API backed by a persistent Claude Code session.

Exposes /v1/chat/completions so the OpenClaw gateway can route agent turns
through a real Claude Code instance with full tool access (Bash, Read, Write,
Edit, Glob, Grep, Agent subprocesses).

The bridge maintains a single persistent ClaudeSDKClient session per
conversation, giving Claude Code full context continuity within a session.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("CLAUDE_BRIDGE_PORT", "18800"))
HOST = os.environ.get("CLAUDE_BRIDGE_HOST", "127.0.0.1")
CWD = os.environ.get("CLAUDE_BRIDGE_CWD", "/root")
MAX_TURNS = int(os.environ.get("CLAUDE_BRIDGE_MAX_TURNS", "25"))
MODEL = os.environ.get("CLAUDE_BRIDGE_MODEL", None)  # None = use default

# Build system prompt from workspace files
WORKSPACE = Path("/root/.openclaw/agents/claude/workspace")
SYSTEM_PROMPT_FILES = ["SOUL.md", "IDENTITY.md", "USER.md", "MEMORY.md", "TOOLS.md", "HEARTBEAT.md", "BOOTSTRAP.md"]

LOG = logging.getLogger("claude-bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    """Read workspace .md files and assemble a system prompt."""
    parts = []
    # Allowed symlink targets — only resolve to known safe directories
    ALLOWED_SYMLINK_PARENTS = [Path("/root/.openclaw"), Path("/root/ClydeMemory")]

    for fname in SYSTEM_PROMPT_FILES:
        fpath = WORKSPACE / fname
        if fpath.is_symlink():
            resolved = fpath.resolve()
            # Prevent symlink traversal to arbitrary files
            if not any(str(resolved).startswith(str(p)) for p in ALLOWED_SYMLINK_PARENTS):
                LOG.warning(f"Symlink {fname} points outside allowed dirs: {resolved} — skipping")
                continue
            fpath = resolved
        if fpath.exists():
            content = fpath.read_text().strip()
            if content:
                parts.append(f"<{fname.replace('.md', '').upper()}>\n{content}\n</{fname.replace('.md', '').upper()}>")

    header = (
        "You are Clyde, Derek's AI assistant. You are running as a Claude Code "
        "instance behind the OpenClaw gateway, accessible via ClawDash (web UI). "
        "You have full Claude Code tool access: Bash, Read, Write, Edit, Glob, Grep, "
        "and Agent subprocesses. Use them freely.\n\n"
        "## MANDATORY MEMORY PROTOCOL — DO THIS BEFORE EVERY RESPONSE\n\n"
        "You have persistent memory stored in a vector database. Before answering "
        "ANY question about projects, infrastructure, past work, clients, configs, "
        "APIs, keys, decisions, or anything Derek has told you before:\n\n"
        "1. FIRST run: `openclaw-memo search \"<relevant keywords>\" --user derek --limit 5`\n"
        "2. Use the results to inform your response\n"
        "3. If Derek shares new facts, run: `openclaw-memo add \"<fact>\" --user derek`\n\n"
        "DO NOT skip this step. DO NOT say \"I don't have\" or \"I can't find\" "
        "without searching memory first. The answer is often already stored.\n"
        "DO NOT announce that you searched — just integrate the results naturally.\n\n"
    )
    return header + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------
class SessionManager:
    """Manages persistent Claude Code sessions keyed by conversation ID."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}  # conv_id -> {client, last_used, messages}
        self._lock = asyncio.Lock()

    async def get_or_create(self, conv_id: str) -> "SessionHandle":
        async with self._lock:
            if conv_id in self.sessions:
                sess = self.sessions[conv_id]
                sess["last_used"] = time.time()
                return SessionHandle(sess, is_new=False)

            sess = {
                "conv_id": conv_id,
                "last_used": time.time(),
                "message_count": 0,
            }
            self.sessions[conv_id] = sess
            return SessionHandle(sess, is_new=True)

    async def cleanup_stale(self, max_age_s: int = 3600):
        """Remove sessions older than max_age_s."""
        async with self._lock:
            now = time.time()
            stale = [k for k, v in self.sessions.items() if now - v["last_used"] > max_age_s]
            for k in stale:
                LOG.info(f"Cleaning up stale session {k}")
                del self.sessions[k]


class SessionHandle:
    def __init__(self, sess: dict, is_new: bool):
        self.sess = sess
        self.is_new = is_new


sessions = SessionManager()

# ---------------------------------------------------------------------------
# Claude Code query wrapper
# ---------------------------------------------------------------------------

# Dangerous command patterns that should never be auto-approved
BLOCKED_COMMANDS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "shutdown", "reboot",
    "halt", "poweroff", "init 0", "init 6",
    "> /dev/sd", "chmod -R 777 /", "chown -R",
    "curl | sh", "curl | bash", "wget | sh", "wget | bash",
    "DROP DATABASE", "DROP TABLE", "TRUNCATE",
    "passwd", "userdel", "deluser",
]

# Paths that should not be written to or deleted
PROTECTED_PATHS = [
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "/root/.ssh/", "/boot/", "/usr/bin/", "/usr/sbin/",
]


async def approve_all_tools(
    tool_name: str, tool_input: dict, context: "ToolPermissionContext"
) -> "PermissionResult":
    """Auto-approve tool calls with safety blocklist for destructive operations."""
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    # Check Bash commands against blocklist
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", "")).strip()
        cmd_lower = cmd.lower()
        for blocked in BLOCKED_COMMANDS:
            if blocked.lower() in cmd_lower:
                LOG.warning(f"BLOCKED tool call: {tool_name} — matched '{blocked}' in: {cmd[:100]}")
                return PermissionResultDeny(reason=f"Blocked: command matches dangerous pattern '{blocked}'")

    # Check file operations against protected paths
    if tool_name in ("Write", "Edit"):
        path = str(tool_input.get("file_path", ""))
        for protected in PROTECTED_PATHS:
            if path.startswith(protected):
                LOG.warning(f"BLOCKED tool call: {tool_name} on protected path: {path}")
                return PermissionResultDeny(reason=f"Blocked: cannot modify protected path '{protected}'")

    return PermissionResultAllow()


async def run_claude_turn(messages: list[dict], conv_id: str) -> str:
    """Run a single turn through Claude Code SDK using ClaudeSDKClient."""
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

    # Extract the last user message
    user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                user_msg = " ".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
            else:
                user_msg = str(content)
            break

    if not user_msg:
        return "No user message found."

    system_prompt = build_system_prompt()

    handle = await sessions.get_or_create(conv_id)
    handle.sess["message_count"] += 1

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        permission_mode="default",
        can_use_tool=approve_all_tools,
        cwd=CWD,
        max_turns=MAX_TURNS,
        continue_conversation=not handle.is_new,
        env={
            "CLAUDE_CONFIG_DIR": "/root/.clydecodebot/claude-auth",
        },
    )

    if MODEL:
        options.model = MODEL

    # Collect response text from the last assistant message
    response_parts = []

    try:
        from claude_agent_sdk import (
            AssistantMessage, ResultMessage, TextBlock,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_msg)

            async for event in client.receive_response():
                cls_name = type(event).__name__

                if isinstance(event, AssistantMessage):
                    content = getattr(event, "content", [])
                    for block in content:
                        if isinstance(block, TextBlock) or hasattr(block, "text"):
                            text = getattr(block, "text", "")
                            if text:
                                response_parts.append(text)

                elif isinstance(event, ResultMessage):
                    # ResultMessage may also carry text
                    text = getattr(event, "text", "")
                    if text:
                        response_parts.append(text)
                    # receive_response() terminates after this

    except Exception as e:
        LOG.error(f"Claude Code error: {e}", exc_info=True)
        return f"Error running Claude Code: {e}"

    result = "\n".join(response_parts).strip()
    if not result:
        result = "(Claude Code completed the turn but produced no text output — likely performed tool actions only.)"

    # Fire-and-forget: store the exchange in memory
    asyncio.create_task(_ingest_turn(user_msg, result, conv_id))

    return result


# ---------------------------------------------------------------------------
# Live memory ingest — runs after each turn
# ---------------------------------------------------------------------------
async def _ingest_turn(user_msg: str, assistant_reply: str, conv_id: str):
    """Extract key facts from a turn and store via openclaw-memo."""
    try:
        # Skip trivial exchanges
        if len(assistant_reply) < 40 or assistant_reply.startswith("Error"):
            return

        # Build a compact summary for fact extraction
        exchange = f"User: {user_msg[:500]}\nAssistant: {assistant_reply[:1500]}"

        # Use a simple heuristic: if the reply contains technical details,
        # decisions, configs, paths, or commands — it's worth storing.
        # We extract a one-line fact from the exchange.
        import subprocess
        proc = await asyncio.create_subprocess_exec(
            "openclaw-memo", "add",
            f"[ClawDash session {conv_id[:8]}] {_extract_fact(user_msg, assistant_reply)}",
            "--user", "derek",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        await asyncio.wait_for(proc.wait(), timeout=15)
        LOG.info(f"Live ingest: stored fact for conv {conv_id[:8]}")
    except asyncio.TimeoutError:
        LOG.warning("Live ingest timed out")
    except Exception as e:
        LOG.warning(f"Live ingest error: {e}")


def _extract_fact(user_msg: str, reply: str) -> str:
    """Extract a concise fact from a user/assistant exchange.

    Heuristic: take the user's question and the first substantive
    sentence of the reply. Max 400 chars for memory storage.
    """
    # Clean up the user message
    q = user_msg.strip().replace("\n", " ")[:150]

    # Get the first meaningful line of the reply (skip empty, headers, code fences)
    answer_lines = []
    for line in reply.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        answer_lines.append(line)
        if len(" ".join(answer_lines)) > 200:
            break

    a = " ".join(answer_lines)[:250]

    fact = f"Q: {q} → A: {a}"
    return fact[:400]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG.info(f"Claude Code Bridge starting on {HOST}:{PORT}")
    LOG.info(f"Workspace: {WORKSPACE}")
    LOG.info(f"CWD: {CWD}")
    # Start periodic cleanup
    async def cleanup_loop():
        while True:
            await asyncio.sleep(600)
            await sessions.cleanup_stale()

    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="Claude Code Bridge", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint backed by Claude Code."""
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Use a conversation ID from the request or generate one
    conv_id = body.get("conversation_id") or body.get("session_id") or "default"

    if stream:
        return await _stream_response(messages, conv_id, body)
    else:
        return await _sync_response(messages, conv_id, body)


async def _sync_response(messages: list[dict], conv_id: str, body: dict) -> JSONResponse:
    """Non-streaming response."""
    text = await run_claude_turn(messages, conv_id)

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "claude-code-bridge",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


async def _stream_response(messages: list[dict], conv_id: str, body: dict):
    """SSE streaming response (OpenAI format)."""
    async def generate():
        text = await run_claude_turn(messages, conv_id)

        # Send as a single chunk (Claude Code doesn't stream token-by-token through SDK)
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "claude-code-bridge",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        # Send finish
        finish = {
            "id": chunk["id"],
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "claude-code-bridge",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(finish)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/models")
async def list_models():
    """Model listing endpoint."""
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": "claude-code-bridge",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openclaw",
        }],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions.sessions)}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
