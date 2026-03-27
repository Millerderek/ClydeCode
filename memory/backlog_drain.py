#!/usr/bin/env python3
"""
backlog_drain.py — Telegram Backlog Drain + Memory Fold

When ClydeCodeBot is offline, Telegram queues messages. On restart the bot
calls drop_pending_updates=True, silently discarding them. This script runs
BEFORE the bot starts (or standalone) to:

  1. Batch-fetch all pending updates via getUpdates
  2. Filter stale/duplicate messages
  3. Group by chat, summarize via OpenRouter (Haiku)
  4. Fold summaries into ClydeMemory via daemon socket
  5. Send a single catch-up reply per chat
  6. ACK updates so the bot's drop_pending won't lose them

Usage:
  backlog_drain.py                   # Run drain (standalone or pre-boot)
  backlog_drain.py --dry-run         # Show what would be drained, don't act
  backlog_drain.py --status          # Show offset + pending count

Can also be imported:
  from backlog_drain import drain_backlog
  drained = drain_backlog()          # Returns count of messages folded
"""

import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OPENROUTER_API_KEY, OPENROUTER_KEY_FILE, CLYDE_STATE_DIR

# ── Config ───────────────────────────────────────────────────────────────────

# Telegram token — from env or .env file
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

TG_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

OFFSET_FILE = CLYDE_STATE_DIR / "tg_drain_offset.json"
LOCK_FILE = Path("/tmp/backlog-drain.lock")

MAX_BACKLOG_AGE = 3600          # Ignore messages older than 1 hour
MIN_BACKLOG_TO_DRAIN = 3        # Only drain if >= N pending messages
MAX_UPDATES_PER_FETCH = 100     # Telegram API max

# LLM summarization
SUMMARIZE_MODEL = "anthropic/claude-haiku-4-5"
MAX_SUMMARY_TOKENS = 800


# ── Offset tracking ─────────────────────────────────────────────────────────

def load_offset() -> int:
    """Load last-processed update_id."""
    if OFFSET_FILE.exists():
        try:
            return json.loads(OFFSET_FILE.read_text()).get("offset", 0)
        except (json.JSONDecodeError, KeyError):
            pass
    # Fallback: check the delivery-queue offset (shared with other scripts)
    fallback = Path("/root/.openclaw/delivery-queue/tg-offset.json")
    if fallback.exists():
        try:
            return json.loads(fallback.read_text()).get("offset", 0)
        except Exception:
            pass
    return 0


def save_offset(offset: int):
    """Persist current offset."""
    OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_FILE.write_text(json.dumps({
        "offset": offset,
        "updated": datetime.now(timezone.utc).isoformat(),
    }))


# ── Telegram API ─────────────────────────────────────────────────────────────

def fetch_pending_updates(offset: int) -> list[dict]:
    """Pull all pending updates without long-polling (timeout=0)."""
    url = f"{TG_BASE}/getUpdates"
    params = json.dumps({
        "offset": offset,
        "limit": MAX_UPDATES_PER_FETCH,
        "timeout": 0,
    }).encode()

    req = urllib.request.Request(
        url, data=params,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("result", [])
    except Exception as e:
        _log(f"getUpdates failed: {e}")
        return []


def send_reply(chat_id: int, text: str):
    """Send a Telegram message."""
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }).encode()
    req = urllib.request.Request(
        f"{TG_BASE}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as e:
        _log(f"sendMessage failed for chat {chat_id}: {e}")


def acknowledge_updates(last_update_id: int):
    """Advance Telegram's server-side offset so bot doesn't reprocess."""
    payload = json.dumps({
        "offset": last_update_id + 1,
        "limit": 1,
        "timeout": 0,
    }).encode()
    req = urllib.request.Request(
        f"{TG_BASE}/getUpdates",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass
    save_offset(last_update_id + 1)


# ── Filtering ────────────────────────────────────────────────────────────────

def filter_backlog(updates: list[dict]) -> list[dict]:
    """Remove stale, non-message, and duplicate updates."""
    now = time.time()
    seen = set()
    valid = []

    for u in updates:
        msg = u.get("message") or u.get("edited_message")
        if not msg:
            continue

        # Age filter
        msg_time = msg.get("date", 0)
        if now - msg_time > MAX_BACKLOG_AGE:
            continue

        text = (msg.get("text") or "").strip()
        if not text:
            continue

        chat_id = msg["chat"]["id"]

        # Deduplicate near-identical messages per chat
        dedup_key = f"{chat_id}:{text[:80]}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        user_info = msg.get("from", {})
        valid.append({
            "update_id": u["update_id"],
            "chat_id": chat_id,
            "user": user_info.get("username") or user_info.get("first_name", "unknown"),
            "text": text,
            "timestamp": msg_time,
        })

    return valid


# ── LLM Summarization ───────────────────────────────────────────────────────

def _get_openrouter_key() -> str:
    """Resolve OpenRouter API key."""
    if OPENROUTER_API_KEY:
        return OPENROUTER_API_KEY
    if OPENROUTER_KEY_FILE.exists():
        for line in OPENROUTER_KEY_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def fold_to_summary(messages: list[dict]) -> dict:
    """
    Summarize a batch of backlog messages via OpenRouter.

    Returns:
        {"memory_summary": str, "reply": str, "dropped": str}
    """
    api_key = _get_openrouter_key()
    if not api_key:
        _log("WARNING: No OpenRouter key — using passthrough mode")
        texts = [m["text"] for m in messages]
        return {
            "memory_summary": f"Backlog ({len(messages)} msgs): " + " | ".join(texts)[:500],
            "reply": f"⚡ Back online. Caught up on {len(messages)} queued messages.",
            "dropped": "",
        }

    formatted = "\n".join([
        f"[{datetime.fromtimestamp(m['timestamp'], tz=timezone.utc).strftime('%H:%M UTC')}] "
        f"@{m['user']}: {m['text']}"
        for m in messages
    ])

    system_prompt = (
        "You are the OpenClaw memory fold worker. "
        "Given a batch of queued Telegram messages sent while the bot was offline, "
        "produce a JSON object with three keys:\n"
        "1. \"memory_summary\": Terse bullet-point summary of distinct intents, "
        "decisions, or context to carry forward. Max 10 bullets. "
        "Focus on actionable items and technical details.\n"
        "2. \"reply\": A brief catch-up reply to the user acknowledging the backlog, "
        "confirming what was understood, and what will be acted on. "
        "Do NOT pretend these were handled in real-time. Be honest about the gap.\n"
        "3. \"dropped\": Any messages that were too stale, vague, or nonsensical "
        "to act on. Empty string if none.\n"
        "Return ONLY valid JSON, no markdown fences."
    )

    payload = json.dumps({
        "model": SUMMARIZE_MODEL,
        "max_tokens": MAX_SUMMARY_TOKENS,
        "messages": [
            {"role": "user", "content": f"Backlog messages:\n\n{formatted}"}
        ],
        "system": system_prompt,
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openclaw/ClydeMemory",
            "X-Title": "ClydeMemory Backlog Drain",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            raw = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log(f"LLM summarization failed: {e}")
        return {
            "memory_summary": f"Backlog ({len(messages)} msgs, summarization failed)",
            "reply": f"⚡ Back online. {len(messages)} queued messages received but summarization failed. I'll review manually.",
            "dropped": "",
        }

    # Parse JSON response
    try:
        # Strip markdown fences if present
        clean = re.sub(r'^```json?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        result = json.loads(clean)
    except json.JSONDecodeError:
        _log(f"JSON parse failed, using raw text. First 200 chars: {raw[:200]}")
        result = {
            "memory_summary": raw[:500],
            "reply": f"⚡ Back online. Caught up on {len(messages)} queued messages.",
            "dropped": "",
        }

    # Ensure all keys exist
    result.setdefault("memory_summary", "")
    result.setdefault("reply", "")
    result.setdefault("dropped", "")

    return result


# ── Memory Storage ───────────────────────────────────────────────────────────

def store_to_memory(summary: str, chat_id: int, msg_count: int):
    """Fold summary into ClydeMemory via daemon socket."""
    from daemon_client import memo_add

    # Prefix with metadata for searchability
    tagged = f"[Telegram backlog drain, chat={chat_id}, {msg_count} msgs] {summary}"

    from config import CLYDE_USER
    success = memo_add(tagged, user_id=CLYDE_USER, impact="normal", writer="telegram-drain")
    if success:
        _log(f"  Stored to memory: {len(tagged)} chars")
    else:
        _log(f"  WARNING: memo_add failed for chat {chat_id}")


# ── Locking ──────────────────────────────────────────────────────────────────

def _acquire_lock() -> bool:
    """Simple file-based lock to prevent bot + drain race."""
    if LOCK_FILE.exists():
        # Check if stale (older than 5 min)
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < 300:
            return False
        _log(f"Removing stale lock (age={age:.0f}s)")
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock():
    try:
        LOCK_FILE.unlink()
    except OSError:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [drain] {msg}", flush=True)


def drain_backlog(dry_run: bool = False) -> int:
    """
    Main drain logic. Returns count of messages folded.

    Can be called from bot startup or standalone.
    """
    if not TELEGRAM_TOKEN:
        _log("ERROR: No TELEGRAM_BOT_TOKEN found")
        return 0

    if not _acquire_lock():
        _log("Another drain is running, skipping")
        return 0

    try:
        return _drain_inner(dry_run)
    finally:
        _release_lock()


def _drain_inner(dry_run: bool) -> int:
    offset = load_offset()
    updates = fetch_pending_updates(offset)

    if not updates:
        _log("No pending updates")
        return 0

    backlog = filter_backlog(updates)
    last_id = updates[-1]["update_id"]

    _log(f"Found {len(updates)} raw updates, {len(backlog)} after filtering")

    if len(backlog) < MIN_BACKLOG_TO_DRAIN:
        _log(f"Below threshold ({MIN_BACKLOG_TO_DRAIN}), passing through to bot")
        if not dry_run:
            save_offset(last_id + 1)
        return 0

    # Group by chat
    by_chat: dict[int, list] = defaultdict(list)
    for msg in backlog:
        by_chat[msg["chat_id"]].append(msg)

    total_folded = 0

    for chat_id, messages in by_chat.items():
        _log(f"Chat {chat_id}: {len(messages)} messages to fold")

        if dry_run:
            for m in messages:
                ts = datetime.fromtimestamp(m["timestamp"], tz=timezone.utc)
                _log(f"  [{ts.strftime('%H:%M')}] @{m['user']}: {m['text'][:80]}")
            total_folded += len(messages)
            continue

        # Summarize
        result = fold_to_summary(messages)

        # Store to memory
        if result["memory_summary"]:
            store_to_memory(result["memory_summary"], chat_id, len(messages))

        # Log dropped messages
        if result.get("dropped"):
            _log(f"  Dropped: {result['dropped'][:200]}")

        # Send catch-up reply
        reply_text = f"🔄 *Backlog drained* ({len(messages)} messages)\n\n{result['reply']}"
        send_reply(chat_id, reply_text)

        total_folded += len(messages)
        _log(f"  Folded {len(messages)} messages, reply sent")

    # ACK all updates (even ones we filtered out)
    if not dry_run:
        acknowledge_updates(last_id)
        _log(f"Offset advanced to {last_id + 1}")

    return total_folded


def show_status():
    """Show current offset and pending count."""
    offset = load_offset()
    updates = fetch_pending_updates(offset)
    backlog = filter_backlog(updates) if updates else []

    print(f"Offset file:     {OFFSET_FILE}")
    print(f"Current offset:  {offset}")
    print(f"Pending updates: {len(updates)}")
    print(f"After filtering: {len(backlog)}")
    print(f"Drain threshold: {MIN_BACKLOG_TO_DRAIN}")

    if backlog:
        print(f"\nPending messages:")
        for m in backlog[:10]:
            ts = datetime.fromtimestamp(m["timestamp"], tz=timezone.utc)
            print(f"  [{ts.strftime('%H:%M')}] @{m['user']}: {m['text'][:80]}")
        if len(backlog) > 10:
            print(f"  ... and {len(backlog) - 10} more")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--status" in args:
        show_status()
    elif "--dry-run" in args:
        count = drain_backlog(dry_run=True)
        print(f"\n[dry-run] Would have drained {count} messages")
    else:
        count = drain_backlog()
        if count:
            _log(f"Done. Drained {count} messages total.")
        else:
            _log("Nothing to drain.")
