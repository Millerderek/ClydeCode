#!/usr/bin/env python3
"""
lcm_engine.py — Lossless Context Management for ClydeMemory.

Archives raw conversation messages to PostgreSQL, builds a hierarchical
summary DAG so context survives compaction/restart, and provides recall
injection + full-text search across all archived history.

Tables:
  lcm_sessions         — session registry (active/compacted/closed)
  lcm_messages         — raw turn archive with FTS via tsvector
  lcm_summary_nodes    — hierarchical summary DAG (leaf → rollup)
  lcm_compaction_events — tracks when context was lost

Daemon integration:
  lcm_search   — FTS across all archived messages
  lcm_context  — compressed history for recall injection

Usage:
  lcm_engine.py migrate              # Create/update tables
  lcm_engine.py archive [--days N]   # Archive messages from JSONL transcripts
  lcm_engine.py summarize            # Build/extend summary DAG
  lcm_engine.py recall <session_id>  # Assemble compressed history for injection
  lcm_engine.py search "query"       # FTS across all history
  lcm_engine.py stats                # Show LCM statistics
  lcm_engine.py cleanup [--days N]   # Prune old messages (keep summaries)
  lcm_engine.py pressure <sid> [--compact]  # Check/trigger compaction
  lcm_engine.py grep "query" [--session ID] # FTS grep across history
  lcm_engine.py describe <sid> [--turns S-E] # Describe session/turn range
"""

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLAUDE_PROJECT_DIRS as PROJECT_DIRS,
    CLYDE_STATE_DIR,
    OPENROUTER_API_KEY,
    OPENROUTER_KEY_FILE,
)
from db import get_pg, pg_execute, pg_query


def _pg_fetchall(sql: str, params=None) -> list[tuple]:
    """Execute SQL and return raw tuples (avoids pipe-delimiter issues with content)."""
    conn = get_pg()
    if conn is None:
        return []
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall() if cur.description else []

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

LEAF_BATCH_TOKENS = 2000       # ~2000 tokens per leaf summary (~8000 chars)
ROLLUP_FAN_IN = 4              # 4 children per parent rollup
FRESH_TAIL_TURNS = 20          # recent turns use leaf (most detailed) summaries
MAX_DEPTH = 4                  # max DAG depth (covers ~512 leaf batches)
RECALL_BUDGET_TOKENS = 1500    # ~6000 chars of compressed history for injection
ARCHIVE_RETENTION_DAYS = 90    # raw message retention (summaries kept forever)
SUMMARIZE_MODEL = "anthropic/claude-haiku-4-5"

# Feature 1: Context-aware compaction trigger
COMPACTION_THRESHOLD = float(os.environ.get("LCM_COMPACTION_THRESHOLD", "0.75"))
MODEL_CONTEXT_LIMIT = int(os.environ.get("LCM_MODEL_CONTEXT_LIMIT", "200000"))

# Feature 3: Large file handler
LARGE_FILE_TOKEN_THRESHOLD = int(os.environ.get("LCM_LARGE_FILE_THRESHOLD", "25000"))

# Also scan the clydecodebot project dirs
_EXTRA_DIRS = [
    Path("/root/.clydecodebot/claude-auth/projects"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Schema migration
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- LCM Session registry
CREATE TABLE IF NOT EXISTS lcm_sessions (
    session_id    TEXT PRIMARY KEY,
    source_file   TEXT,
    started_at    TIMESTAMPTZ DEFAULT now(),
    last_turn     INTEGER DEFAULT 0,
    last_archive_at TIMESTAMPTZ,
    status        TEXT DEFAULT 'active'
);

-- LCM Raw message archive
CREATE TABLE IF NOT EXISTS lcm_messages (
    id            SERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES lcm_sessions(session_id),
    turn_index    INTEGER NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    token_est     INTEGER,
    ts            TIMESTAMPTZ DEFAULT now(),
    UNIQUE(session_id, turn_index)
);
CREATE INDEX IF NOT EXISTS idx_lcm_msg_session
    ON lcm_messages(session_id, turn_index);

-- FTS index on messages (GIN on tsvector)
-- We use a generated column approach via trigger since GENERATED ALWAYS
-- requires PG12+ and some Docker images lag behind.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'lcm_messages' AND column_name = 'tsv'
    ) THEN
        ALTER TABLE lcm_messages ADD COLUMN tsv TSVECTOR;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_lcm_msg_fts ON lcm_messages USING GIN(tsv);

-- Trigger to auto-update tsv on insert/update
CREATE OR REPLACE FUNCTION lcm_messages_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS lcm_messages_tsv_update ON lcm_messages;
CREATE TRIGGER lcm_messages_tsv_update
    BEFORE INSERT OR UPDATE OF content ON lcm_messages
    FOR EACH ROW EXECUTE FUNCTION lcm_messages_tsv_trigger();

-- LCM Summary DAG
CREATE TABLE IF NOT EXISTS lcm_summary_nodes (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    depth         INTEGER NOT NULL,
    parent_id     TEXT REFERENCES lcm_summary_nodes(id),
    summary       TEXT NOT NULL,
    token_est     INTEGER,
    turn_start    INTEGER NOT NULL,
    turn_end      INTEGER NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_lcm_dag_session
    ON lcm_summary_nodes(session_id, depth, turn_start);

-- LCM Compaction events
CREATE TABLE IF NOT EXISTS lcm_compaction_events (
    id            SERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL,
    detected_at   TIMESTAMPTZ DEFAULT now(),
    turns_before  INTEGER,
    turns_after   INTEGER
);
"""


def migrate():
    """Create or update LCM tables."""
    conn = get_pg()
    if conn is None:
        print("ERROR: Cannot connect to PostgreSQL")
        return False

    cur = conn.cursor()

    # Execute each statement individually to handle $$ blocks properly.
    # Split on lines starting with a known SQL keyword or $$ delimiter.
    statements = [
        # 1. Sessions table
        """CREATE TABLE IF NOT EXISTS lcm_sessions (
            session_id    TEXT PRIMARY KEY,
            source_file   TEXT,
            started_at    TIMESTAMPTZ DEFAULT now(),
            last_turn     INTEGER DEFAULT 0,
            last_archive_at TIMESTAMPTZ,
            status        TEXT DEFAULT 'active'
        )""",
        # 2. Messages table
        """CREATE TABLE IF NOT EXISTS lcm_messages (
            id            SERIAL PRIMARY KEY,
            session_id    TEXT NOT NULL REFERENCES lcm_sessions(session_id),
            turn_index    INTEGER NOT NULL,
            role          TEXT NOT NULL,
            content       TEXT NOT NULL,
            token_est     INTEGER,
            ts            TIMESTAMPTZ DEFAULT now(),
            tsv           TSVECTOR,
            UNIQUE(session_id, turn_index)
        )""",
        # 3. Indexes
        "CREATE INDEX IF NOT EXISTS idx_lcm_msg_session ON lcm_messages(session_id, turn_index)",
        "CREATE INDEX IF NOT EXISTS idx_lcm_msg_fts ON lcm_messages USING GIN(tsv)",
        # 4. Trigger function (single $$ block — must be executed as one statement)
        """CREATE OR REPLACE FUNCTION lcm_messages_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql""",
        # 5. Drop + create trigger
        "DROP TRIGGER IF EXISTS lcm_messages_tsv_update ON lcm_messages",
        """CREATE TRIGGER lcm_messages_tsv_update
            BEFORE INSERT OR UPDATE OF content ON lcm_messages
            FOR EACH ROW EXECUTE FUNCTION lcm_messages_tsv_trigger()""",
        # 6. Summary DAG table
        """CREATE TABLE IF NOT EXISTS lcm_summary_nodes (
            id            TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            depth         INTEGER NOT NULL,
            parent_id     TEXT REFERENCES lcm_summary_nodes(id),
            summary       TEXT NOT NULL,
            token_est     INTEGER,
            turn_start    INTEGER NOT NULL,
            turn_end      INTEGER NOT NULL,
            created_at    TIMESTAMPTZ DEFAULT now()
        )""",
        "CREATE INDEX IF NOT EXISTS idx_lcm_dag_session ON lcm_summary_nodes(session_id, depth, turn_start)",
        # 7. Compaction events table
        """CREATE TABLE IF NOT EXISTS lcm_compaction_events (
            id            SERIAL PRIMARY KEY,
            session_id    TEXT NOT NULL,
            detected_at   TIMESTAMPTZ DEFAULT now(),
            turns_before  INTEGER,
            turns_after   INTEGER
        )""",
        # 8. Large file storage (Feature 3)
        """CREATE TABLE IF NOT EXISTS lcm_large_files (
            id            SERIAL PRIMARY KEY,
            session_id    TEXT NOT NULL,
            turn_index    INTEGER NOT NULL,
            file_hint     TEXT,
            content       TEXT NOT NULL,
            summary       TEXT,
            token_estimate INTEGER,
            created_at    TIMESTAMPTZ DEFAULT NOW()
        )""",
        "CREATE INDEX IF NOT EXISTS idx_lcm_large_files_session ON lcm_large_files(session_id, turn_index)",
    ]

    for stmt in statements:
        try:
            cur.execute(stmt)
        except Exception as e:
            err = str(e).lower()
            if "already exists" not in err:
                print(f"  WARNING: {e}")
            conn.rollback()

    print("LCM schema migration complete.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# JSONL parsing (reused from conversation_digest.py, extended for LCM)
# ═══════════════════════════════════════════════════════════════════════════════

SKIP_PREFIXES = (
    "<system-reminder",
    "<RELEVANT_CONTEXT",
    "<CLYDE_MEMORY",
    "<function_calls>",
    "<OPENCLAW_MEMORY>",
    "<available-deferred-tools>",
)


def _parse_jsonl_turns(path: Path) -> list[dict]:
    """
    Parse a JSONL transcript into a list of turn dicts:
      [{"role": "user"|"assistant"|"tool_use"|"tool_result", "content": "...", "turn_index": N}]

    Keeps FULL content (not truncated like conversation_digest).
    """
    turns = []
    idx = 0

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                t = obj.get("type", "")
                msg = obj.get("message", {})

                if t == "user":
                    content = msg.get("content", "")
                    text = content if isinstance(content, str) else ""
                    if text and not any(text.startswith(p) for p in SKIP_PREFIXES):
                        turns.append({
                            "role": "user",
                            "content": text,
                            "turn_index": idx,
                        })
                        idx += 1

                elif t == "assistant":
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
                        turns.append({
                            "role": "assistant",
                            "content": text,
                            "turn_index": idx,
                        })
                        idx += 1

    except Exception as e:
        print(f"  WARNING: Error parsing {path}: {e}")

    return turns


def _session_id_from_path(path: Path) -> str:
    """Extract session ID from JSONL filename (UUID portion)."""
    return path.stem  # e.g., "1f2450d3-d73e-44c7-8f48-30f41bfbe412"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 1: Context-aware compaction trigger
# ═══════════════════════════════════════════════════════════════════════════════

def check_context_pressure(session_id: str) -> dict:
    """
    Check how much of the model context window is consumed by unsummarized
    messages in a session. Returns {ratio, tokens, limit, needs_compaction}.
    """
    raw = pg_query(
        """SELECT COALESCE(SUM(token_est), 0) FROM lcm_messages
           WHERE session_id = %s
             AND turn_index > COALESCE(
                 (SELECT MAX(turn_end) FROM lcm_summary_nodes
                  WHERE session_id = %s AND depth = 0), -1
             )""",
        (session_id, session_id)
    )
    unsummarized_tokens = int(raw.strip()) if raw and raw.strip() else 0

    ratio = unsummarized_tokens / MODEL_CONTEXT_LIMIT if MODEL_CONTEXT_LIMIT > 0 else 0
    return {
        "ratio": round(ratio, 4),
        "tokens": unsummarized_tokens,
        "limit": MODEL_CONTEXT_LIMIT,
        "needs_compaction": ratio >= COMPACTION_THRESHOLD,
    }


def compact_if_needed(session_id: str) -> dict:
    """
    Check context pressure and auto-compact if threshold exceeded.
    Returns compaction result or None if not needed.
    """
    pressure = check_context_pressure(session_id)
    if not pressure["needs_compaction"]:
        return {"compacted": False, **pressure}

    # Run leaf + rollup summaries
    leaves = build_leaf_summaries(session_id)
    rollups = build_rollup_summaries(session_id)

    # Log compaction event
    pg_execute(
        """INSERT INTO lcm_compaction_events (session_id, turns_before, turns_after)
           VALUES (%s, %s, %s)""",
        (session_id, pressure["tokens"],
         check_context_pressure(session_id)["tokens"])
    )

    return {
        "compacted": True,
        "leaves_created": leaves,
        "rollups_created": rollups,
        "pressure_before": pressure["ratio"],
        "pressure_after": check_context_pressure(session_id)["ratio"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 3: Large file detection helpers
# ═══════════════════════════════════════════════════════════════════════════════

_FILE_MARKERS = (
    "{", "[", "#!", "---", "<?xml", "<!DOCTYPE", "<!doctype",
    "CREATE ", "ALTER ", "SELECT ", "INSERT ", "DROP ",
    "package ", "import ", "from ", "def ", "class ", "func ",
    "module ", "namespace ", "using ", "#include",
)


def _is_file_like(content: str) -> bool:
    """Detect if content looks like a file rather than conversation."""
    stripped = content.strip()

    # Check for common file markers
    if any(stripped.startswith(m) for m in _FILE_MARKERS):
        return True

    # Check dialogue density — real conversation has User:/Assistant: lines
    lines = stripped.splitlines()
    if not lines:
        return False
    dialogue_lines = sum(1 for l in lines
                         if l.strip().startswith(("User:", "Assistant:", "Human:")))
    dialogue_ratio = dialogue_lines / len(lines) if lines else 0

    # Low dialogue ratio + large = probably a file
    return dialogue_ratio < 0.05


def _detect_file_hint(content: str) -> str:
    """Try to detect a filename from the content."""
    # Look for common patterns like "# filename.py" or "// filename.js"
    first_lines = content[:500].splitlines()[:5]
    for line in first_lines:
        line = line.strip()
        # Shebang
        if line.startswith("#!"):
            return line.split("/")[-1] if "/" in line else "script"
        # Comment with filename
        for prefix in ("# ", "// ", "/* ", "-- "):
            if line.startswith(prefix):
                rest = line[len(prefix):].strip()
                if "." in rest and len(rest) < 60 and " " not in rest:
                    return rest
    # YAML front matter
    if content.strip().startswith("---"):
        return "config.yaml"
    # JSON
    if content.strip().startswith("{") or content.strip().startswith("["):
        return "data.json"
    return None


def summarize_large_file(file_id: int) -> str:
    """Generate and cache a summary for a large file. Returns the summary."""
    rows = _pg_fetchall(
        "SELECT content, summary FROM lcm_large_files WHERE id = %s",
        (file_id,)
    )
    if not rows:
        return ""

    content, existing_summary = rows[0]
    if existing_summary:
        return existing_summary

    # Generate summary via LLM
    # Truncate content for summarization (send first + last chunks)
    if len(content) > 20_000:
        truncated = content[:10_000] + "\n\n[...middle truncated...]\n\n" + content[-5_000:]
    else:
        truncated = content

    summary = _llm_summarize(
        f"Summarize this file/document concisely. Focus on structure, "
        f"key components, and purpose:\n\n{truncated}",
        "aggressive"
    )

    # Cache the summary
    pg_execute(
        "UPDATE lcm_large_files SET summary = %s WHERE id = %s",
        (summary, file_id)
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Archive — write raw messages to PG
# ═══════════════════════════════════════════════════════════════════════════════

def find_transcripts(days: int = 7) -> list[Path]:
    """Find JSONL transcripts modified in last N days."""
    cutoff = time.time() - days * 86400
    files = []
    all_dirs = list(PROJECT_DIRS) + _EXTRA_DIRS

    for d in all_dirs:
        if not d.exists():
            continue
        # Recurse into subdirs (projects have nested dirs)
        for f in d.rglob("*.jsonl"):
            # Skip subagent transcripts
            if "/subagents/" in str(f):
                continue
            try:
                if f.stat().st_mtime > cutoff:
                    files.append(f)
            except OSError:
                continue

    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return files


def archive_session(path: Path) -> int:
    """
    Archive messages from a single JSONL transcript to PG.
    Returns number of new messages archived.
    """
    session_id = _session_id_from_path(path)

    # Ensure session exists
    existing = pg_query(
        "SELECT last_turn FROM lcm_sessions WHERE session_id = %s",
        (session_id,)
    )

    if existing:
        last_turn = int(existing.strip()) if existing.strip() else 0
    else:
        pg_execute(
            "INSERT INTO lcm_sessions (session_id, source_file, status) VALUES (%s, %s, 'active')",
            (session_id, str(path))
        )
        last_turn = 0

    # Parse all turns
    turns = _parse_jsonl_turns(path)
    if not turns:
        return 0

    # Only archive turns we haven't seen
    new_turns = [t for t in turns if t["turn_index"] >= last_turn]
    if not new_turns:
        return 0

    conn = get_pg()
    if conn is None:
        return 0

    cur = conn.cursor()
    archived = 0

    for turn in new_turns:
        content = turn["content"]
        token_est = _estimate_tokens(content)

        # Feature 3: Divert large files to separate storage
        if token_est > LARGE_FILE_TOKEN_THRESHOLD and _is_file_like(content):
            try:
                file_hint = _detect_file_hint(content)
                cur.execute(
                    """INSERT INTO lcm_large_files
                       (session_id, turn_index, file_hint, content, token_estimate)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT DO NOTHING""",
                    (session_id, turn["turn_index"], file_hint, content, token_est)
                )
                # Replace content with stub in messages table
                content = (f"[Large file stored separately: turn={turn['turn_index']}, "
                           f"~{token_est} tokens, file={file_hint or 'unknown'}]")
                token_est = _estimate_tokens(content)
            except Exception as e:
                print(f"    WARNING: Large file divert failed: {e}")
                conn.rollback()
                # Fall through to normal archiving with truncation
                if len(content) > 50_000:
                    content = content[:50_000] + "\n[...truncated]"
        elif len(content) > 50_000:
            # Cap individual message content at 50KB to prevent bloat
            content = content[:50_000] + "\n[...truncated]"

        try:
            cur.execute(
                """INSERT INTO lcm_messages (session_id, turn_index, role, content, token_est)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (session_id, turn_index) DO NOTHING""",
                (session_id, turn["turn_index"], turn["role"],
                 content, token_est)
            )
            archived += 1
        except Exception as e:
            # Log but don't fail the whole batch
            if "duplicate" not in str(e).lower():
                print(f"    WARNING: {e}")
            conn.rollback()
            continue

    # Update session last_turn
    max_turn = max(t["turn_index"] for t in turns)
    pg_execute(
        """UPDATE lcm_sessions SET last_turn = %s, last_archive_at = now()
           WHERE session_id = %s AND last_turn < %s""",
        (max_turn, session_id, max_turn)
    )

    return archived


def archive_all(days: int = 7) -> dict:
    """Archive messages from all recent transcripts."""
    files = find_transcripts(days)
    stats = {"files": 0, "messages": 0, "skipped": 0}

    for path in files:
        try:
            count = archive_session(path)
            if count > 0:
                session_id = _session_id_from_path(path)
                print(f"  [{session_id[:12]}] +{count} messages")
                stats["messages"] += count
                stats["files"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            print(f"  WARNING: {path.name}: {e}")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Summarize — build hierarchical DAG
# ═══════════════════════════════════════════════════════════════════════════════

def _get_openrouter_key() -> str:
    """Get OPENROUTER_API_KEY from config or fallback to key file."""
    if OPENROUTER_API_KEY:
        return OPENROUTER_API_KEY
    if OPENROUTER_KEY_FILE.exists():
        for line in OPENROUTER_KEY_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _llm_summarize(text: str, level: str = "normal") -> str:
    """
    Call LLM to summarize text.
    level: "normal" (default), "aggressive" (bullet points), "deterministic" (no LLM)
    """
    if level == "deterministic":
        # Fallback: first/last sentence of each paragraph
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) <= 3:
            return text
        kept = [lines[0], lines[len(lines) // 2], lines[-1]]
        result = " | ".join(kept)
        return result[:2000]

    api_key = _get_openrouter_key()
    if not api_key:
        return _llm_summarize(text, "deterministic")

    prompts = {
        "normal": (
            "Summarize this conversation segment concisely. "
            "Preserve: decisions made, technical details, action items, "
            "corrections, and any state changes. "
            "Output a brief paragraph (3-5 sentences max).\n\n"
            f"{text}"
        ),
        "aggressive": (
            "Compress this conversation into key bullet points. "
            "Only keep: decisions, technical facts, action items, errors/fixes. "
            "Use telegraphic style. Max 5 bullets.\n\n"
            f"{text}"
        ),
    }

    prompt = prompts.get(level, prompts["normal"])

    payload = json.dumps({
        "model": SUMMARIZE_MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Millerderek/ClydeMemory",
            "X-Title": "ClydeMemory LCM Summarizer",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  WARNING: LLM summarize failed ({e}) — falling back to deterministic")
        return _llm_summarize(text, "deterministic")


def _get_unsummarized_ranges(session_id: str) -> list[tuple[int, int]]:
    """
    Find turn ranges in a session that have archived messages but no leaf summary.
    Returns list of (start, end) tuples, each covering ~LEAF_BATCH_TOKENS tokens.
    """
    # Get all archived turns for this session
    raw = pg_query(
        """SELECT turn_index, token_est FROM lcm_messages
           WHERE session_id = %s ORDER BY turn_index""",
        (session_id,)
    )
    if not raw:
        return []

    turns = []
    for line in raw.strip().splitlines():
        parts = line.split("|")
        turns.append((int(parts[0]), int(parts[1] or 0)))

    # Get existing leaf summaries
    raw_leaves = pg_query(
        """SELECT turn_start, turn_end FROM lcm_summary_nodes
           WHERE session_id = %s AND depth = 0 ORDER BY turn_start""",
        (session_id,)
    )
    covered = set()
    if raw_leaves:
        for line in raw_leaves.strip().splitlines():
            parts = line.split("|")
            start, end = int(parts[0]), int(parts[1])
            for i in range(start, end + 1):
                covered.add(i)

    # Find uncovered turns, batch by token count
    ranges = []
    batch_start = None
    batch_tokens = 0

    for turn_idx, token_est in turns:
        if turn_idx in covered:
            continue

        if batch_start is None:
            batch_start = turn_idx

        batch_tokens += token_est

        if batch_tokens >= LEAF_BATCH_TOKENS:
            ranges.append((batch_start, turn_idx))
            batch_start = None
            batch_tokens = 0

    # Don't summarize the very latest turns (they're still in active context)
    # Leave the last FRESH_TAIL_TURNS unsummarized
    if turns:
        max_turn = turns[-1][0]
        cutoff = max_turn - FRESH_TAIL_TURNS
        ranges = [(s, e) for s, e in ranges if e < cutoff]

    return ranges


def build_leaf_summaries(session_id: str) -> int:
    """Build depth-0 (leaf) summaries for unsummarized turn ranges."""
    ranges = _get_unsummarized_ranges(session_id)
    if not ranges:
        return 0

    created = 0
    for start, end in ranges:
        # Fetch the actual messages (use tuple fetch to avoid pipe issues)
        rows = _pg_fetchall(
            """SELECT role, content FROM lcm_messages
               WHERE session_id = %s AND turn_index BETWEEN %s AND %s
               ORDER BY turn_index""",
            (session_id, start, end)
        )
        if not rows:
            continue

        # Build transcript
        lines = []
        for role, content in rows:
            # Truncate very long individual messages for summarization
            if len(content) > 2000:
                content = content[:2000] + "..."
            lines.append(f"{role.capitalize()}: {content}")

        transcript = "\n".join(lines)
        summary = _llm_summarize(transcript, "normal")

        node_id = f"leaf-{session_id[:8]}-{start}-{end}"
        token_est = _estimate_tokens(summary)

        try:
            pg_execute(
                """INSERT INTO lcm_summary_nodes
                   (id, session_id, depth, summary, token_est, turn_start, turn_end)
                   VALUES (%s, %s, 0, %s, %s, %s, %s)
                   ON CONFLICT (id) DO NOTHING""",
                (node_id, session_id, summary, token_est, start, end)
            )
            created += 1
        except Exception as e:
            print(f"  WARNING: Failed to create leaf {node_id}: {e}")

    return created


def build_rollup_summaries(session_id: str) -> int:
    """Build higher-depth rollup summaries from orphan nodes."""
    created = 0

    for depth in range(1, MAX_DEPTH + 1):
        child_depth = depth - 1

        # Find orphan nodes at child_depth (no parent)
        rows = _pg_fetchall(
            """SELECT id, summary, turn_start, turn_end FROM lcm_summary_nodes
               WHERE session_id = %s AND depth = %s AND parent_id IS NULL
               ORDER BY turn_start""",
            (session_id, child_depth)
        )
        if not rows:
            break

        orphans = []
        for node_id, summary, turn_start, turn_end in rows:
            orphans.append({
                "id": node_id,
                "summary": summary or "",
                "turn_start": turn_start,
                "turn_end": turn_end,
            })

        # Need at least ROLLUP_FAN_IN orphans to create a rollup
        while len(orphans) >= ROLLUP_FAN_IN:
            batch = orphans[:ROLLUP_FAN_IN]
            orphans = orphans[ROLLUP_FAN_IN:]

            # Combine summaries
            combined = "\n---\n".join(o["summary"] for o in batch)
            rollup_summary = _llm_summarize(combined, "aggressive")

            turn_start = batch[0]["turn_start"]
            turn_end = batch[-1]["turn_end"]
            node_id = f"roll-{depth}-{session_id[:8]}-{turn_start}-{turn_end}"
            token_est = _estimate_tokens(rollup_summary)

            try:
                pg_execute(
                    """INSERT INTO lcm_summary_nodes
                       (id, session_id, depth, summary, token_est, turn_start, turn_end)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                    (node_id, session_id, depth, rollup_summary, token_est, turn_start, turn_end)
                )

                # Set parent on children
                for child in batch:
                    pg_execute(
                        "UPDATE lcm_summary_nodes SET parent_id = %s WHERE id = %s",
                        (node_id, child["id"])
                    )

                created += 1
            except Exception as e:
                print(f"  WARNING: Failed to create rollup {node_id}: {e}")

    return created


def summarize_session(session_id: str) -> dict:
    """Build all summary levels for a session."""
    leaves = build_leaf_summaries(session_id)
    rollups = build_rollup_summaries(session_id)
    return {"leaves": leaves, "rollups": rollups}


def summarize_all() -> dict:
    """Build summaries for all active sessions."""
    raw = pg_query("SELECT session_id FROM lcm_sessions WHERE status = 'active'")
    if not raw:
        return {"sessions": 0, "leaves": 0, "rollups": 0}

    stats = {"sessions": 0, "leaves": 0, "rollups": 0}
    for line in raw.strip().splitlines():
        session_id = line.strip()
        if not session_id:
            continue

        result = summarize_session(session_id)
        if result["leaves"] > 0 or result["rollups"] > 0:
            print(f"  [{session_id[:12]}] +{result['leaves']} leaves, +{result['rollups']} rollups")
            stats["sessions"] += 1
            stats["leaves"] += result["leaves"]
            stats["rollups"] += result["rollups"]

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Recall — assemble compressed history for injection
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_recall(session_id: str, budget_tokens: int = RECALL_BUDGET_TOKENS) -> str:
    """
    Assemble compressed session history for recall injection.

    Strategy:
    - Recent turns: use leaf summaries (most detail)
    - Older turns: use highest-depth rollup available
    - Fills up to budget_tokens
    """
    # Get all summary nodes for this session, ordered by turn range
    rows = _pg_fetchall(
        """SELECT id, depth, summary, token_est, turn_start, turn_end
           FROM lcm_summary_nodes
           WHERE session_id = %s
           ORDER BY turn_start DESC, depth DESC""",
        (session_id,)
    )
    if not rows:
        return ""

    nodes = []
    for node_id, depth, summary, token_est, turn_start, turn_end in rows:
        nodes.append({
            "id": node_id,
            "depth": depth,
            "summary": summary,
            "token_est": token_est or 0,
            "turn_start": turn_start,
            "turn_end": turn_end,
        })

    if not nodes:
        return ""

    # Greedy selection: prefer higher depth (more compressed) for old turns,
    # lower depth (more detailed) for recent turns
    selected = []
    covered_ranges = set()  # track which turn ranges we've covered
    tokens_used = 0

    for node in nodes:
        # Skip if this range overlaps with something we already selected
        node_range = set(range(node["turn_start"], node["turn_end"] + 1))
        if node_range & covered_ranges:
            continue

        if tokens_used + node["token_est"] > budget_tokens:
            continue

        selected.append(node)
        covered_ranges |= node_range
        tokens_used += node["token_est"]

    if not selected:
        return ""

    # Sort by turn order for coherent reading
    selected.sort(key=lambda n: n["turn_start"])

    # Format
    parts = ["## Session History (compressed)"]
    for node in selected:
        depth_label = "leaf" if node["depth"] == 0 else f"L{node['depth']}"
        parts.append(f"[turns {node['turn_start']}-{node['turn_end']} ({depth_label})]")
        parts.append(node["summary"])
        parts.append("")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Search — FTS across all archived messages
# ═══════════════════════════════════════════════════════════════════════════════

def search_history(query: str, session_id: str = None, limit: int = 10) -> list[dict]:
    """Full-text search across archived conversation messages."""
    # Sanitize query for tsquery
    words = re.findall(r'\w+', query)
    if not words:
        return []
    tsquery = " & ".join(words[:8])  # max 8 terms

    if session_id:
        rows = _pg_fetchall(
            """SELECT m.session_id, m.turn_index, m.role, m.content,
                      ts_rank(m.tsv, to_tsquery('english', %s)) AS rank
               FROM lcm_messages m
               WHERE m.session_id = %s AND m.tsv @@ to_tsquery('english', %s)
               ORDER BY rank DESC
               LIMIT %s""",
            (tsquery, session_id, tsquery, limit)
        )
    else:
        rows = _pg_fetchall(
            """SELECT m.session_id, m.turn_index, m.role, m.content,
                      ts_rank(m.tsv, to_tsquery('english', %s)) AS rank
               FROM lcm_messages m
               WHERE m.tsv @@ to_tsquery('english', %s)
               ORDER BY rank DESC
               LIMIT %s""",
            (tsquery, tsquery, limit)
        )

    results = []
    for sid, turn_idx, role, content, rank in rows:
        results.append({
            "session_id": sid,
            "turn_index": turn_idx,
            "role": role,
            "content": content[:500],  # truncate for display
            "rank": float(rank),
        })

    return results


def get_context(session_id: str, from_turn: int, to_turn: int,
                mode: str = "summary") -> str:
    """
    Get context for a turn range.
    mode="summary": return best available summary
    mode="raw": return raw messages
    """
    if mode == "raw":
        rows = _pg_fetchall(
            """SELECT role, content FROM lcm_messages
               WHERE session_id = %s AND turn_index BETWEEN %s AND %s
               ORDER BY turn_index""",
            (session_id, from_turn, to_turn)
        )
        if not rows:
            return ""
        lines = []
        for role, content in rows:
            lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines)

    # Summary mode: find best covering summary node
    rows = _pg_fetchall(
        """SELECT summary FROM lcm_summary_nodes
           WHERE session_id = %s
             AND turn_start <= %s AND turn_end >= %s
           ORDER BY depth DESC
           LIMIT 1""",
        (session_id, from_turn, to_turn)
    )
    if rows:
        return rows[0][0]

    # No covering summary — fall back to raw
    return get_context(session_id, from_turn, to_turn, mode="raw")


# ═══════════════════════════════════════════════════════════════════════════════
# Maintenance
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup(retention_days: int = ARCHIVE_RETENTION_DAYS) -> dict:
    """Prune old archived messages. Keep summaries forever."""
    result = pg_query(
        """DELETE FROM lcm_messages
           WHERE ts < now() - interval '%s days'
           RETURNING id""",
        (retention_days,)
    )
    deleted = len(result.strip().splitlines()) if result else 0

    # Mark sessions with no remaining messages as closed
    pg_execute(
        """UPDATE lcm_sessions SET status = 'closed'
           WHERE session_id NOT IN (SELECT DISTINCT session_id FROM lcm_messages)
             AND status != 'closed'"""
    )

    return {"deleted_messages": deleted, "retention_days": retention_days}


def stats() -> dict:
    """Return LCM statistics."""
    s = {}

    raw = pg_query("SELECT count(*), COALESCE(sum(token_est), 0) FROM lcm_messages")
    if raw:
        parts = raw.strip().split("|")
        s["total_messages"] = int(parts[0])
        s["total_tokens"] = int(parts[1])

    raw = pg_query("SELECT count(*) FROM lcm_sessions WHERE status = 'active'")
    s["active_sessions"] = int(raw.strip()) if raw else 0

    raw = pg_query("SELECT count(*) FROM lcm_summary_nodes")
    s["summary_nodes"] = int(raw.strip()) if raw else 0

    raw = pg_query("SELECT count(*) FROM lcm_summary_nodes WHERE depth = 0")
    s["leaf_summaries"] = int(raw.strip()) if raw else 0

    raw = pg_query("SELECT count(*) FROM lcm_summary_nodes WHERE depth > 0")
    s["rollup_summaries"] = int(raw.strip()) if raw else 0

    raw = pg_query("SELECT max(depth) FROM lcm_summary_nodes")
    s["max_depth"] = int(raw.strip()) if raw and raw.strip() else 0

    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Daemon handler functions (imported by memo_daemon.py)
# ═══════════════════════════════════════════════════════════════════════════════

def handle_lcm_search(params: dict) -> dict:
    """Daemon handler: search archived conversation history."""
    query = params.get("query", "")
    session_id = params.get("session_id")
    limit = params.get("limit", 10)

    if not query:
        return {"ok": False, "error": "query is required"}

    results = search_history(query, session_id=session_id, limit=limit)
    return {"ok": True, "results": results, "count": len(results)}


def handle_lcm_context(params: dict) -> dict:
    """Daemon handler: get context for a session (summary or raw)."""
    session_id = params.get("session_id", "")
    mode = params.get("mode", "summary")  # "summary" or "raw"

    if not session_id:
        return {"ok": False, "error": "session_id is required"}

    if mode == "recall":
        # Full compressed recall assembly
        budget = params.get("budget_tokens", RECALL_BUDGET_TOKENS)
        text = assemble_recall(session_id, budget_tokens=budget)
        return {"ok": True, "text": text, "tokens": _estimate_tokens(text)}

    from_turn = params.get("from_turn", 0)
    to_turn = params.get("to_turn", 999999)
    text = get_context(session_id, from_turn, to_turn, mode=mode)
    return {"ok": True, "text": text, "tokens": _estimate_tokens(text)}


def handle_lcm_stats(params: dict) -> dict:
    """Daemon handler: LCM statistics."""
    return {"ok": True, **stats()}


# ── Feature 1: Compaction trigger handler ─────────────────────────────────────

def handle_lcm_pressure(params: dict) -> dict:
    """Daemon handler: check context pressure and optionally compact."""
    session_id = params.get("session_id", "")
    auto_compact = params.get("auto_compact", False)

    if not session_id:
        return {"ok": False, "error": "session_id is required"}

    pressure = check_context_pressure(session_id)

    if auto_compact and pressure["needs_compaction"]:
        result = compact_if_needed(session_id)
        return {"ok": True, **result}

    return {"ok": True, **pressure}


# ── Feature 2: Agent-facing history tools ─────────────────────────────────────

def handle_lcm_grep(params: dict) -> dict:
    """
    Daemon handler: FTS grep across all sessions.
    Returns ranked snippets with session ID + turn info.
    """
    query = params.get("query", "")
    session_id = params.get("session_id")
    limit = params.get("limit", 10)

    if not query:
        return {"ok": False, "error": "query is required"}

    results = search_history(query, session_id=session_id, limit=limit)

    # Also search large files
    words = re.findall(r'\w+', query)
    if words:
        tsquery = " & ".join(words[:8])
        try:
            lf_rows = _pg_fetchall(
                """SELECT id, session_id, turn_index, file_hint,
                          ts_rank(to_tsvector('english', content),
                                  to_tsquery('english', %s)) AS rank
                   FROM lcm_large_files
                   WHERE to_tsvector('english', content) @@ to_tsquery('english', %s)
                   ORDER BY rank DESC LIMIT 5""",
                (tsquery, tsquery)
            )
            for fid, sid, tidx, hint, rank in lf_rows:
                results.append({
                    "session_id": sid,
                    "turn_index": tidx,
                    "role": "large_file",
                    "content": f"[Large file: {hint or 'unknown'}, id={fid}]",
                    "rank": float(rank),
                    "large_file_id": fid,
                })
        except Exception:
            pass  # FTS on large files is best-effort

    return {"ok": True, "results": results, "count": len(results)}


def handle_lcm_describe(params: dict) -> dict:
    """
    Daemon handler: describe a session or turn range.
    Returns the best available summary, walking DAG upward.
    """
    session_id = params.get("session_id", "")
    start_turn = params.get("start_turn", 0)
    end_turn = params.get("end_turn", 999999)

    if not session_id:
        return {"ok": False, "error": "session_id is required"}

    # Try to find overlapping summary nodes (prefer highest depth)
    rows = _pg_fetchall(
        """SELECT id, depth, summary, token_est, turn_start, turn_end
           FROM lcm_summary_nodes
           WHERE session_id = %s
             AND turn_start <= %s AND turn_end >= %s
           ORDER BY depth DESC, turn_start ASC
           LIMIT 5""",
        (session_id, end_turn, start_turn)
    )

    if rows:
        summaries = []
        for node_id, depth, summary, token_est, ts, te in rows:
            summaries.append({
                "node_id": node_id,
                "depth": depth,
                "summary": summary,
                "turn_range": [ts, te],
            })
        return {"ok": True, "summaries": summaries}

    # No covering summary — check if there are large files in range
    lf_rows = _pg_fetchall(
        """SELECT id, turn_index, file_hint, summary
           FROM lcm_large_files
           WHERE session_id = %s AND turn_index BETWEEN %s AND %s""",
        (session_id, start_turn, end_turn)
    )
    large_files = []
    for fid, tidx, hint, summary in lf_rows:
        if not summary:
            summary = summarize_large_file(fid)
        large_files.append({
            "file_id": fid,
            "turn_index": tidx,
            "file_hint": hint,
            "summary": summary,
        })

    # Fall back to raw messages
    text = get_context(session_id, start_turn, end_turn, mode="raw")
    result = {"ok": True, "raw_context": text[:4000]}
    if large_files:
        result["large_files"] = large_files
    return result


def handle_lcm_recall(params: dict) -> dict:
    """
    Daemon handler: assemble token-budgeted context from history.
    On-demand recall assembly callable by agent mid-conversation.
    """
    query = params.get("query", "")
    budget = params.get("token_budget", 4000)
    session_id = params.get("session_id")

    if not query:
        return {"ok": False, "error": "query is required"}

    # If session_id given, do session-specific recall
    if session_id:
        text = assemble_recall(session_id, budget_tokens=budget)
        if text:
            return {"ok": True, "text": text, "tokens": _estimate_tokens(text),
                    "source": "session_recall"}

    # Cross-session: FTS to find relevant sessions, then assemble from best matches
    results = search_history(query, limit=20)
    if not results:
        return {"ok": True, "text": "", "tokens": 0, "source": "none"}

    # Group by session, pick top sessions
    session_scores = {}
    for r in results:
        sid = r["session_id"]
        session_scores[sid] = session_scores.get(sid, 0) + r["rank"]

    top_sessions = sorted(session_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    # Assemble recall from top sessions within budget
    parts = []
    tokens_used = 0
    per_session_budget = budget // len(top_sessions)

    for sid, score in top_sessions:
        recall = assemble_recall(sid, budget_tokens=per_session_budget)
        if recall:
            est = _estimate_tokens(recall)
            if tokens_used + est <= budget:
                parts.append(f"### Session {sid[:12]} (relevance: {score:.2f})\n{recall}")
                tokens_used += est

    text = "\n\n".join(parts) if parts else ""
    return {"ok": True, "text": text, "tokens": tokens_used,
            "source": "cross_session", "sessions_used": len(parts)}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "migrate":
        migrate()

    elif cmd == "archive":
        days = 7
        if "--days" in sys.argv:
            idx = sys.argv.index("--days")
            if idx + 1 < len(sys.argv):
                days = int(sys.argv[idx + 1])
        print(f"Archiving messages from last {days} days...")
        result = archive_all(days)
        print(f"\nDone. {result['files']} file(s), {result['messages']} message(s) archived, "
              f"{result['skipped']} skipped.")

    elif cmd == "summarize":
        print("Building summary DAG...")
        result = summarize_all()
        print(f"\nDone. {result['sessions']} session(s), "
              f"{result['leaves']} leaves, {result['rollups']} rollups created.")

    elif cmd == "recall":
        if len(sys.argv) < 3:
            print("Usage: lcm_engine.py recall <session_id>")
            sys.exit(1)
        session_id = sys.argv[2]
        text = assemble_recall(session_id)
        if text:
            print(text)
        else:
            print("No compressed history available for this session.")

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: lcm_engine.py search \"query\"")
            sys.exit(1)
        query = sys.argv[2]
        results = search_history(query, limit=10)
        if not results:
            print("No results.")
        else:
            for r in results:
                print(f"  [{r['session_id'][:12]}] turn {r['turn_index']} ({r['role']}): "
                      f"{r['content'][:120]}...")

    elif cmd == "stats":
        s = stats()
        print(f"\n  LCM Statistics")
        print(f"  {'=' * 40}")
        print(f"  Active sessions:   {s.get('active_sessions', 0)}")
        print(f"  Total messages:    {s.get('total_messages', 0)}")
        print(f"  Est. tokens:       {s.get('total_tokens', 0):,}")
        print(f"  Summary nodes:     {s.get('summary_nodes', 0)}")
        print(f"    Leaves (depth 0): {s.get('leaf_summaries', 0)}")
        print(f"    Rollups:          {s.get('rollup_summaries', 0)}")
        print(f"    Max depth:        {s.get('max_depth', 0)}")

    elif cmd == "cleanup":
        days = ARCHIVE_RETENTION_DAYS
        if "--days" in sys.argv:
            idx = sys.argv.index("--days")
            if idx + 1 < len(sys.argv):
                days = int(sys.argv[idx + 1])
        result = cleanup(days)
        print(f"Cleaned up {result['deleted_messages']} message(s) older than {days} days.")

    elif cmd == "pressure":
        # Feature 1: Check context pressure for a session
        if len(sys.argv) < 3:
            print("Usage: lcm_engine.py pressure <session_id> [--compact]")
            sys.exit(1)
        session_id = sys.argv[2]
        auto_compact = "--compact" in sys.argv
        if auto_compact:
            result = compact_if_needed(session_id)
            if result.get("compacted"):
                print(f"Compacted: {result['leaves_created']} leaves, "
                      f"{result['rollups_created']} rollups")
                print(f"Pressure: {result['pressure_before']:.1%} → {result['pressure_after']:.1%}")
            else:
                print(f"No compaction needed. Pressure: {result['ratio']:.1%}")
        else:
            p = check_context_pressure(session_id)
            print(f"Session: {session_id[:12]}")
            print(f"Unsummarized tokens: {p['tokens']:,}")
            print(f"Context limit: {p['limit']:,}")
            print(f"Pressure: {p['ratio']:.1%}")
            print(f"Needs compaction: {p['needs_compaction']}")

    elif cmd == "grep":
        # Feature 2: FTS grep
        if len(sys.argv) < 3:
            print("Usage: lcm_engine.py grep \"query\" [--session ID] [--limit N]")
            sys.exit(1)
        query = sys.argv[2]
        sid = None
        limit = 10
        if "--session" in sys.argv:
            idx = sys.argv.index("--session")
            if idx + 1 < len(sys.argv):
                sid = sys.argv[idx + 1]
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit")
            if idx + 1 < len(sys.argv):
                limit = int(sys.argv[idx + 1])
        result = handle_lcm_grep({"query": query, "session_id": sid, "limit": limit})
        if not result.get("results"):
            print("No results.")
        else:
            for r in result["results"]:
                role = r.get("role", "?")
                print(f"  [{r['session_id'][:12]}] turn {r['turn_index']} ({role}) "
                      f"rank={r['rank']:.3f}")
                print(f"    {r['content'][:150]}")

    elif cmd == "describe":
        # Feature 2: Describe a session/turn range
        if len(sys.argv) < 3:
            print("Usage: lcm_engine.py describe <session_id> [--turns START-END]")
            sys.exit(1)
        session_id = sys.argv[2]
        start, end = 0, 999999
        if "--turns" in sys.argv:
            idx = sys.argv.index("--turns")
            if idx + 1 < len(sys.argv):
                parts = sys.argv[idx + 1].split("-")
                start = int(parts[0])
                end = int(parts[1]) if len(parts) > 1 else start
        result = handle_lcm_describe({
            "session_id": session_id, "start_turn": start, "end_turn": end
        })
        if result.get("summaries"):
            for s in result["summaries"]:
                print(f"  [{s['node_id']}] depth={s['depth']} turns={s['turn_range']}")
                print(f"    {s['summary'][:300]}")
        elif result.get("raw_context"):
            print(result["raw_context"][:2000])
        if result.get("large_files"):
            for lf in result["large_files"]:
                print(f"  [Large file: {lf['file_hint']}] turn={lf['turn_index']}")
                print(f"    {lf['summary'][:200]}")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
