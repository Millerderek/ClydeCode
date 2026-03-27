#!/usr/bin/env python3
"""
Gateway Session Digest — periodic ingest of ClawDash conversation history.

Reads gateway session JSONL files for the claude agent, extracts key facts
from user/assistant exchanges, and stores them via openclaw-memo.

Tracks last-processed position per session file to avoid re-ingesting.
State stored in .digest_state.json alongside this script.

Run via cron every 30 minutes:
  */30 * * * * /usr/bin/python3 /root/.openclaw/agents/claude/bridge/digest_sessions.py >> /tmp/claude-digest.log 2>&1
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SESSIONS_DIR = Path("/root/.openclaw/agents/claude/sessions")
STATE_FILE = Path(__file__).parent / ".digest_state.json"
USER = "derek"
MAX_FACTS_PER_RUN = 20  # Cap to avoid flooding memory
MIN_REPLY_LEN = 50       # Skip trivial replies
MAX_FACT_LEN = 400

LOG_PREFIX = "[claude-digest]"


def log(msg: str):
    print(f"{datetime.now().isoformat()} {LOG_PREFIX} {msg}")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# JSONL parser
# ---------------------------------------------------------------------------
def parse_session_messages(filepath: Path, after_line: int = 0) -> list[dict]:
    """Parse user/assistant message pairs from a gateway session JSONL file."""
    exchanges = []
    current_user = None
    line_num = 0

    with open(filepath) as f:
        for line in f:
            line_num += 1
            if line_num <= after_line:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") != "message":
                continue

            msg = entry.get("message", {})
            role = msg.get("role")
            content = msg.get("content", "")

            # Extract text from content blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                text = " ".join(text_parts)
            elif isinstance(content, str):
                text = content
            else:
                continue

            text = text.strip()
            if not text:
                continue

            if role == "user":
                # Strip the timestamp prefix if present: [Thu 2026-03-26 18:17 EDT]
                text = re.sub(r"^\[.*?\]\s*", "", text)
                current_user = {"text": text, "line": line_num}
            elif role == "assistant" and current_user:
                if len(text) >= MIN_REPLY_LEN:
                    exchanges.append({
                        "user": current_user["text"],
                        "assistant": text,
                        "line": line_num,
                    })
                current_user = None

    return exchanges


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------
def extract_fact(user_msg: str, reply: str) -> str | None:
    """Extract a concise fact from an exchange. Returns None if not worth storing."""
    # Skip greetings, meta-questions
    skip_patterns = [
        r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure)\b",
        r"^(who are you|what can you do|help)\b",
    ]
    user_lower = user_msg.lower().strip()
    for pat in skip_patterns:
        if re.match(pat, user_lower):
            return None

    # Get the user question (trimmed)
    q = user_msg.strip().replace("\n", " ")[:150]

    # Get first substantive lines of the reply
    answer_lines = []
    for line in reply.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            if answer_lines:  # Stop at first code block if we have content
                break
            continue
        # Skip lines that are just formatting
        if line.startswith("|") or line.startswith("---"):
            continue
        answer_lines.append(line)
        if len(" ".join(answer_lines)) > 250:
            break

    if not answer_lines:
        return None

    a = " ".join(answer_lines)[:250]
    fact = f"[ClawDash] Q: {q} → A: {a}"
    return fact[:MAX_FACT_LEN]


def store_fact(fact: str) -> bool:
    """Store a fact via openclaw-memo. Returns True on success."""
    try:
        result = subprocess.run(
            ["openclaw-memo", "add", fact, "--user", USER],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception) as e:
        log(f"Error storing fact: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    state = load_state()
    stored_count = 0
    total_exchanges = 0

    # Find all session JSONL files
    session_files = list(SESSIONS_DIR.glob("*.jsonl"))
    if not session_files:
        log("No session files found")
        return

    log(f"Processing {len(session_files)} session file(s)")

    for filepath in session_files:
        session_id = filepath.stem
        last_line = state.get(session_id, {}).get("last_line", 0)

        exchanges = parse_session_messages(filepath, after_line=last_line)
        if not exchanges:
            continue

        total_exchanges += len(exchanges)
        log(f"Session {session_id[:12]}: {len(exchanges)} new exchange(s) after line {last_line}")

        for ex in exchanges:
            if stored_count >= MAX_FACTS_PER_RUN:
                log(f"Hit max facts per run ({MAX_FACTS_PER_RUN})")
                break

            fact = extract_fact(ex["user"], ex["assistant"])
            if fact:
                if store_fact(fact):
                    stored_count += 1

            # Always update state to the latest line processed
            state[session_id] = {"last_line": ex["line"], "updated": datetime.now().isoformat()}

        if stored_count >= MAX_FACTS_PER_RUN:
            break

    save_state(state)
    log(f"Done: {stored_count} facts stored from {total_exchanges} exchanges")


if __name__ == "__main__":
    main()
