#!/usr/bin/env python3
"""
compaction_watcher — Hot-ingest recently updated Claude Code sessions.

Monitors Claude Code JSONL files for recent activity (modified in last 30 min)
and immediately extracts facts from those sessions via Haiku. This closes the
gap where context compaction fires mid-session before conversation_digest runs.

Runs every 5 minutes via cron. Maintains its own state separate from
conversation_digest.py to avoid double-counting.

Usage:
  compaction_watcher.py          # Process recently active sessions
  compaction_watcher.py --status # Show watcher state
  compaction_watcher.py --force  # Reprocess all sessions modified today
"""

import json
import os
import re
import sys
import time
from pathlib import Path
import httpx

# Import topic classifier from context_gate
sys.path.insert(0, str(Path(__file__).parent))
try:
    from context_gate import classify_topic
except ImportError:
    def classify_topic(text: str) -> str:
        return "general"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from config import CLAUDE_PROJECT_DIRS, CLYDE_STATE_DIR, CLYDE_USER as USER, OPENROUTER_API_KEY, OPENROUTER_KEY_FILE

PROJECT_DIR = CLAUDE_PROJECT_DIRS[0] if CLAUDE_PROJECT_DIRS else Path.home() / ".claude/projects"
STATE_FILE  = CLYDE_STATE_DIR / ".compaction_watcher_state.json"
MAX_SESSIONS_PER_RUN = 10

# Hot-watch window: sessions modified within this many seconds
HOT_WINDOW_SECONDS = 30 * 60  # 30 minutes
TAIL_BYTES = 200_000           # bytes to read from end of large active sessions


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"processed": {}}   # {session_id: {"mtime": float, "tail_hash": str, "facts_stored": int}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# JSONL parsing — reuse conversation_digest logic
# ---------------------------------------------------------------------------
def parse_tail(path: Path) -> str:
    """
    Extract a readable transcript from recent portions of a JSONL.
    Reuses conversation_digest's parsing approach for consistency.
    """
    size = path.stat().st_size
    turns = []

    skip_prefixes = (
        "<system-reminder",
        "<RELEVANT_CONTEXT",
        "<CLYDE_MEMORY",
        "<function_calls>",
        "<available-deferred",
        "<task-notification",
    )

    try:
        if size > TAIL_BYTES:
            with open(path, "rb") as f:
                f.seek(-TAIL_BYTES, 2)
                raw = f.read().decode("utf-8", errors="replace")
            lines = raw.splitlines()[1:]  # skip potentially cut first line
        else:
            lines = path.read_text(errors="replace").splitlines()

        for line in lines:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue

            entry_type = e.get("type", "")
            msg = e.get("message", {})
            if not isinstance(msg, dict):
                continue

            content = msg.get("content", "")

            if entry_type == "user":
                if isinstance(content, str):
                    text = content.strip()
                    # Strip injected context blocks, keep real user text
                    if any(text.startswith(p) for p in skip_prefixes):
                        # Try to extract real message after the injection block
                        for tag_end in ["</RELEVANT_CONTEXT>\n", "</system-reminder>\n", "</CLYDE_MEMORY>\n"]:
                            if tag_end in text:
                                remainder = text.split(tag_end)[-1].strip()
                                if len(remainder) > 10:
                                    text = remainder
                                    break
                        else:
                            text = ""
                    if text and len(text) > 10:
                        turns.append(f"User: {text[:400]}")
                elif isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            t = b.get("text", "").strip()
                            if t and len(t) > 10:
                                turns.append(f"User: {t[:400]}")
                            break

            elif entry_type == "assistant":
                if isinstance(content, list):
                    text = next(
                        (b["text"] for b in content
                         if isinstance(b, dict) and b.get("type") == "text"),
                        ""
                    )
                else:
                    text = str(content).strip() if content else ""
                if text and len(text) > 20:
                    turns.append(f"Assistant: {text[:500]}")

    except Exception as e:
        print(f"  WARNING: Error parsing {path.name}: {e}")
        return ""

    return "\n".join(turns[-50:])


# ---------------------------------------------------------------------------
# Fact extraction via OpenRouter (Haiku)
# ---------------------------------------------------------------------------
def get_openrouter_key() -> str:
    if OPENROUTER_API_KEY:
        return OPENROUTER_API_KEY
    if OPENROUTER_KEY_FILE.exists():
        for line in OPENROUTER_KEY_FILE.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def extract_facts_from_summary(summary_text: str) -> list[str]:
    """
    Use Haiku to extract 3-6 concrete facts from a compaction summary.
    Returns list of fact strings ready for clyde-memo add.
    """
    api_key = get_openrouter_key()
    if not api_key:
        print("  WARNING: No OPENROUTER_API_KEY — falling back to regex extraction")
        return _regex_extract(summary_text)

    prompt = f"""Extract 3-6 specific, concrete facts from this conversation summary that should be stored in long-term memory.

Focus on:
- Technical decisions made (file paths, configs, commands, fixes)
- Problems solved and their root causes
- Device/service details (IPs, hostnames, entity names, versions)
- Project state (what's done, what's pending)

Rules:
- Each fact must be a single, self-contained sentence
- Copy ALL proper nouns, file paths, IPs, versions, and numbers VERBATIM
- Skip meta-commentary ("the user asked", "the assistant explained")
- Skip anything already obvious from context

Return ONLY a JSON array of strings. No explanation.

Summary:
{summary_text[:4000]}"""

    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "anthropic/claude-3.5-haiku",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Parse JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            return [f for f in facts if isinstance(f, str) and len(f) > 20]
    except Exception as e:
        print(f"  WARNING: Haiku extraction failed: {e} — falling back to regex")

    return _regex_extract(summary_text)


def _regex_extract(text: str) -> list[str]:
    """Fallback: extract sentences with IPs, file paths, or technical keywords."""
    facts = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        s = s.strip()
        if len(s) < 30:
            continue
        # Score by technical content
        score = 0
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', s):  # IP
            score += 3
        if re.search(r'/[a-z][\w/.-]{3,}', s):  # file path
            score += 2
        if re.search(r'\b(fixed|deployed|configured|created|added|resolved)\b', s, re.I):
            score += 2
        if re.search(r'\b(error|issue|bug|problem)\b', s, re.I):
            score += 1
        if score >= 2:
            facts.append(s)
        if len(facts) >= 5:
            break
    return facts


# ---------------------------------------------------------------------------
# Store facts via clyde-memo
# ---------------------------------------------------------------------------
def store_facts(facts: list[str]) -> int:
    """Store facts via clyde-memo CLI, tagged by topic. Returns count stored."""
    stored = 0
    for fact in facts:
        fact_clean = fact.replace('"', '\\"').replace('\n', ' ').strip()
        if not fact_clean:
            continue
        topic = classify_topic(fact_clean)
        # Prefix fact with topic tag for downstream retrieval scoring
        tagged = f"[{topic}] {fact_clean}" if topic != "general" else fact_clean
        tagged_escaped = tagged.replace('"', '\\"')
        ret = os.system(f'clyde-memo add "{tagged_escaped}" --user {USER} > /dev/null 2>&1')
        if ret == 0:
            stored += 1
            print(f"  OK [{topic}] {fact_clean[:70]}")
        else:
            print(f"  FAIL: Failed to store: {fact_clean[:60]}")
    return stored


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def tail_hash(path: Path) -> str:
    """Cheap hash of last 4KB to detect new content."""
    import hashlib
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > 4096:
                f.seek(-4096, 2)
            return hashlib.md5(f.read()).hexdigest()[:12]
    except Exception:
        return ""


def cmd_process(force: bool = False):
    state = load_state()
    processed = state.get("processed", {})

    # Find recently active JSONL files
    if force:
        cutoff = time.time() - 86400  # today
    else:
        cutoff = time.time() - HOT_WINDOW_SECONDS

    sessions = sorted(
        [f for f in PROJECT_DIR.glob("*.jsonl") if f.stat().st_mtime > cutoff],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )[:MAX_SESSIONS_PER_RUN]

    if not sessions:
        print(f"  No sessions active in last {HOT_WINDOW_SECONDS // 60} minutes")
        return

    total_stored = 0
    processed_count = 0

    for path in sessions:
        sid = path.stem
        mtime = path.stat().st_mtime
        thash = tail_hash(path)
        prev = processed.get(sid, {})

        if not force and prev.get("tail_hash") == thash:
            continue  # no new content since last run

        transcript = parse_tail(path)
        if len(transcript) < 200:
            processed[sid] = {"mtime": mtime, "tail_hash": thash, "facts_stored": 0}
            continue

        print(f"\n  [{path.name[:40]}] -- {len(transcript)} chars of recent activity")
        facts = extract_facts_from_summary(transcript)
        print(f"  Extracted {len(facts)} facts")

        stored = store_facts(facts)
        total_stored += stored
        processed_count += 1

        processed[sid] = {
            "mtime": mtime,
            "tail_hash": thash,
            "facts_stored": stored,
            "processed_at": time.time(),
        }

    state["processed"] = processed
    save_state(state)

    print(f"\n  Sessions processed: {processed_count}")
    print(f"  Facts stored: {total_stored}")


def cmd_status():
    state = load_state()
    processed = state.get("processed", {})
    with_facts = [(sid, v) for sid, v in processed.items() if v.get("facts_stored", 0) > 0]
    total_facts = sum(v.get("facts_stored", 0) for _, v in processed.items())
    print(f"  Sessions tracked: {len(processed)}")
    print(f"  Sessions with extracted facts: {len(with_facts)}")
    print(f"  Total facts extracted: {total_facts}")
    if with_facts:
        print("\n  Recent sessions with facts:")
        for sid, v in sorted(with_facts, key=lambda x: x[1].get("processed_at", 0), reverse=True)[:5]:
            print(f"    {sid[:36]}  facts={v['facts_stored']}")


if __name__ == "__main__":
    if "--status" in sys.argv:
        cmd_status()
    elif "--force" in sys.argv:
        cmd_process(force=True)
    else:
        cmd_process()
