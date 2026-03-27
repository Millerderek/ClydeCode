#!/usr/bin/env python3
"""
conversation_digest — Extract facts from Claude Code JSONL sessions into ClydeMemory.

Reads JSONL logs from Claude Code project directories.
Processes sessions modified in last 7 days (configurable).
Extracts patterns/preferences/decisions via OpenRouter API (claude-haiku).
Stores facts via clyde-memo, with deduplication.

Usage:
  conversation_digest.py              # Process new/changed sessions
  conversation_digest.py --force      # Re-process all sessions in window
  conversation_digest.py --status     # Show ingestion status
  conversation_digest.py --days N     # Override day window (default: 7)
  conversation_digest.py --max N      # Limit sessions per run (default: 20)
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daemon_client import memo_add, memo_search
from topic_classifier import classify_and_store_fact

# ── Config ───────────────────────────────────────────────────────────────────

from config import CLAUDE_PROJECT_DIRS as PROJECT_DIRS
from config import CLYDE_STATE_DIR
from config import CLYDE_USER as USER
from config import OPENROUTER_API_KEY, OPENROUTER_KEY_FILE

STATE_FILE = CLYDE_STATE_DIR / ".digest_state.json"

LARGE_FILE_THRESHOLD = 500_000   # bytes — seek to tail for larger files
TAIL_BYTES = 60_000              # bytes to read from end of large files
MIN_TRANSCRIPT_CHARS = 200       # skip sessions shorter than this
MAX_FACTS_PER_SESSION = 4
DEDUP_SCORE_THRESHOLD = 0.85     # skip if similar memory already exists

EXTRACTION_PROMPT = """You are analyzing a conversation between a user and their AI assistant.

Transcript (recent context):
{transcript}

Extract 0-{max_facts} facts worth remembering. Focus on:
- Technical decisions or preferences the user expressed
- Project state changes (new deployments, configs, file paths, fixes completed)
- Corrections the user made to the assistant
- Non-obvious tooling or workflow preferences specific to this setup

Return ONLY a JSON array of strings. If nothing is worth saving, return [].

Be very selective — skip:
- Generic Linux/IT knowledge
- Anything obvious or widely known
- Greetings, acknowledgements, or meta-conversation about the assistant

Good examples of what TO save:
- "myapp serves static build on port 5100, proxied by toolkit-nginx Docker container"
- "myapp.example.com SSL cert: /etc/letsencrypt/live/myapp.example.com/"
- "Prefers inline CORS middleware over installing cors npm package to avoid Docker rebuild cycles"
- "ClydeCron handles scheduled tasks at ~/.clyde-memory/clydecron/"
- "myapp uses claude-haiku for post-response memory extraction"
"""


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"processed": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Session discovery ─────────────────────────────────────────────────────────

def find_sessions(days: int = 7, max_sessions: int = 20) -> list:
    cutoff = time.time() - days * 86400
    files = []
    for d in PROJECT_DIRS:
        if d.exists():
            files.extend(f for f in d.glob("*.jsonl") if f.stat().st_mtime > cutoff)
    # Most recently modified first
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files[:max_sessions]


# ── Parsing ───────────────────────────────────────────────────────────────────

def _process_jsonl_line(line: str, turns: list):
    """Parse a single JSONL line and append to turns list."""
    line = line.strip()
    if not line:
        return
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return

    t = obj.get("type", "")

    if t == "user":
        msg = obj.get("message", {})
        content = msg.get("content", "")
        text = content if isinstance(content, str) else ""
        # Skip platform-injected context blocks (not real user messages)
        skip_prefixes = (
            "<system-reminder",
            "<RELEVANT_CONTEXT",
            "<CLYDE_MEMORY",
            "<function_calls>",
        )
        if text and not any(text.startswith(p) for p in skip_prefixes):
            turns.append(f"User: {text[:300]}")

    elif t == "assistant":
        msg = obj.get("message", {})
        content = msg.get("content", [])
        if isinstance(content, list):
            text = next(
                (b["text"] for b in content
                 if isinstance(b, dict) and b.get("type") == "text"),
                ""
            )
        else:
            text = str(content) if content else ""
        if text:
            turns.append(f"Assistant: {text[:300]}")


def parse_session(path: Path) -> str:
    """
    Parse a JSONL session file into a clean transcript string.
    For large files, reads only the last TAIL_BYTES to save memory.
    Returns the last 4000 chars of the assembled transcript.
    """
    size = path.stat().st_size
    turns = []

    if size > LARGE_FILE_THRESHOLD:
        # Seek near the end, skip the first (likely partial) line
        with open(path, "rb") as f:
            f.seek(max(0, size - TAIL_BYTES))
            raw = f.read()
        lines = raw.decode("utf-8", errors="replace").splitlines()
        for line in lines[1:]:  # skip potentially cut first line
            _process_jsonl_line(line, turns)
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                _process_jsonl_line(line, turns)

    return "\n".join(turns)[-4000:]


# ── Deduplication ─────────────────────────────────────────────────────────────

def is_duplicate(fact: str) -> bool:
    """Check if a similar fact already exists in memory via daemon search."""
    results = memo_search(fact[:80], user_id=USER, limit=1, skip_gate=True)
    if results:
        score = results[0].get("final", 0)
        return score >= DEDUP_SCORE_THRESHOLD
    return False


# ── Extraction ────────────────────────────────────────────────────────────────

def get_openrouter_key() -> str:
    """Get OPENROUTER_API_KEY from config or fallback to key file."""
    if OPENROUTER_API_KEY:
        return OPENROUTER_API_KEY
    if OPENROUTER_KEY_FILE.exists():
        for line in OPENROUTER_KEY_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def extract_facts(transcript: str) -> list:
    """Call OpenRouter (claude-haiku) to extract learnable facts from transcript."""
    api_key = get_openrouter_key()
    if not api_key:
        print("  WARNING: OPENROUTER_API_KEY not set — skipping extraction")
        return []

    prompt = EXTRACTION_PROMPT.format(
        transcript=transcript,
        max_facts=MAX_FACTS_PER_SESSION
    )

    payload = json.dumps({
        "model": "anthropic/claude-haiku-4-5",
        "max_tokens": 400,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openclaw/ClydeMemory",
            "X-Title": "ClydeMemory Conversation Digest",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"].strip()
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            facts = json.loads(match.group())
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, str) and len(f.strip()) > 20]
    except Exception as e:
        print(f"  WARNING: API error: {e}")

    return []


# ── Storage ───────────────────────────────────────────────────────────────────

def store_fact(fact: str) -> bool:
    """Store a single fact via daemon socket (falls back to CLI)."""
    return memo_add(fact, user_id=USER, writer="digest")


# ── Core processing ───────────────────────────────────────────────────────────

def process_session(path: Path, state: dict, force: bool = False) -> int:
    """
    Process a single session file.
    Returns: -1 = already up to date, 0 = skipped/no facts, N = facts stored
    """
    key = path.name
    current_mtime = path.stat().st_mtime

    if not force and key in state["processed"]:
        if state["processed"][key] == current_mtime:
            return -1  # unchanged since last run

    transcript = parse_session(path)

    if len(transcript) < MIN_TRANSCRIPT_CHARS:
        # Too short to be useful — mark done, no API call
        state["processed"][key] = current_mtime
        return 0

    facts = extract_facts(transcript)
    stored = 0

    for fact in facts:
        fact = fact.strip()
        if not fact:
            continue
        if is_duplicate(fact):
            print(f"    skip (duplicate): {fact[:70]}...")
            continue
        if store_fact(fact):
            print(f"    stored: {fact[:80]}")
            stored += 1
            # Also classify and store in topic system
            try:
                tr = classify_and_store_fact(fact, source=f"digest:{key[:8]}")
                if tr.get("stored"):
                    print(f"      -> topic: {tr['topic_slug']} ({tr['method']})")
                elif tr.get("fact_id") == -1:
                    pass  # duplicate in topic, that's fine
            except Exception as e:
                print(f"      WARNING: topic error: {e}")

    # Mark processed only after successful extraction
    state["processed"][key] = current_mtime
    return stored


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_process(days: int = 7, max_sessions: int = 20, force: bool = False):
    state = load_state()
    sessions = find_sessions(days, max_sessions)

    if not sessions:
        print(f"  No sessions found in last {days} day(s) across {len(PROJECT_DIRS)} dir(s)")
        return

    if not force:
        to_process = [
            s for s in sessions
            if s.name not in state["processed"]
            or state["processed"][s.name] != s.stat().st_mtime
        ]
        if not to_process:
            print(f"  Nothing new to process ({len(sessions)} session(s) already up to date)")
            return
    else:
        to_process = sessions

    print(f"  Processing {len(to_process)} session(s) from last {days}d...\n")

    total_stored = 0
    total_skipped = 0

    for path in to_process:
        size_kb = path.stat().st_size // 1024
        tag = "tail" if path.stat().st_size > LARGE_FILE_THRESHOLD else "full"
        print(f"  [{path.name[:36]}] {size_kb}KB [{tag}]")

        count = process_session(path, state, force=force)

        if count == -1:
            print("    . already up to date")
        elif count == 0:
            print("    . too short or no facts extracted")
            total_skipped += 1
        else:
            print(f"    -> {count} fact(s) stored")
            total_stored += count

        save_state(state)  # save after each session — partial progress is safe

    print(f"\n  Done. {total_stored} fact(s) stored, {total_skipped} session(s) skipped.")

    # Run topic compaction after digest (Tier 2)
    if total_stored > 0:
        try:
            from topic_compactor import run_compaction
            print("\n  Running topic compaction...")
            run_compaction(summarize=True, merge=True, clean=True)
        except Exception as e:
            print(f"  WARNING: Compaction error: {e}")


def cmd_status(days: int = 7, max_sessions: int = 999):
    state = load_state()
    sessions = find_sessions(days, max_sessions)

    print(f"\n  ClydeMemory Conversation Digest -- Status")
    print(f"  {'=' * 50}")
    print(f"  Dirs: {[str(d) for d in PROJECT_DIRS]}")
    print(f"  Sessions in last {days}d: {len(sessions)}")
    print()

    processed = pending = 0
    for s in sessions:
        key = s.name
        mtime = s.stat().st_mtime
        size_kb = s.stat().st_size // 1024
        if key in state["processed"] and state["processed"][key] == mtime:
            status = "OK"
            processed += 1
        else:
            status = "."
            pending += 1
        print(f"  {status} {s.name[:36]}  {size_kb:>6}KB")

    print(f"\n  Processed: {processed}   Pending: {pending}")
    print(f"  Total facts in state: {len(state.get('processed', {}))}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    force  = "--force"  in args
    status = "--status" in args

    days = 7
    if "--days" in args:
        idx = args.index("--days")
        if idx + 1 < len(args):
            try:
                days = int(args[idx + 1])
            except ValueError:
                pass

    max_sessions = 20
    if "--max" in args:
        idx = args.index("--max")
        if idx + 1 < len(args):
            try:
                max_sessions = int(args[idx + 1])
            except ValueError:
                pass

    if status:
        cmd_status(days, max_sessions)
    else:
        cmd_process(days, max_sessions, force)
