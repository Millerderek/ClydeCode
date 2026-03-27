#!/usr/bin/env python3
"""
compaction_trigger.py — Density-based compaction trigger for OpenClaw memory.

Scans topic clusters for memory density. When a topic has too many raw memories
without a recent compaction, triggers compaction (archive old → summarize → store digest).

Designed to run on cron (every 2h) or on-demand.

Phase 1 of OpenClaw Cognitive Architecture.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Trigger compaction when a topic has this many uncompacted memories
DENSITY_THRESHOLD = 12

# Minimum age (hours) since last compaction before re-triggering
COOLDOWN_HOURS = 24

# Redis key prefix for cooldown tracking
REDIS_COOLDOWN_PREFIX = "compaction:cooldown:"

# Max topics to compact per run
MAX_COMPACTIONS_PER_RUN = 3

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _log(msg: str):
    print(f"[compaction_trigger] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Density scan
# ═══════════════════════════════════════════════════════════════════════════════

def get_dense_topics() -> list:
    """
    Find topics with memory count >= DENSITY_THRESHOLD that haven't been
    compacted recently.

    Returns list of dicts: {topic, count}
    """
    # Get memory counts per entity (proxy for topic clustering)
    # Group by the most common entity tags to find dense clusters
    rows = db.pg_query(
        "SELECT me.entity_name, COUNT(DISTINCT me.memory_id) as cnt "
        "FROM memory_entities me "
        "JOIN memories m ON m.id = me.memory_id "
        "WHERE m.is_deprecated = FALSE "
        "  AND m.source != 'compaction' "
        "GROUP BY me.entity_name "
        "HAVING COUNT(DISTINCT me.memory_id) >= %s "
        "ORDER BY cnt DESC "
        "LIMIT 20;",
        (DENSITY_THRESHOLD,)
    )

    if not rows:
        return []

    topics = []
    for line in rows.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 2:
            topics.append({
                "topic": parts[0].strip(),
                "count": int(parts[1].strip()),
            })
    return topics


def check_cooldown(topic: str) -> bool:
    """Check if a topic is still in cooldown (recently compacted)."""
    key = f"{REDIS_COOLDOWN_PREFIX}{topic}"
    result = db.redis_cmd("GET", key)
    return bool(result)


def set_cooldown(topic: str):
    """Set cooldown for a topic after compaction."""
    key = f"{REDIS_COOLDOWN_PREFIX}{topic}"
    ttl_seconds = int(COOLDOWN_HOURS * 3600)
    r = db.get_redis()
    if r:
        r.set(key, str(int(time.time())), ex=ttl_seconds)


# ═══════════════════════════════════════════════════════════════════════════════
# Compaction
# ═══════════════════════════════════════════════════════════════════════════════

def get_memories_for_topic(topic: str, limit: int = 30) -> list:
    """Get all non-compacted memories for a topic entity."""
    rows = db.pg_query(
        "SELECT m.qdrant_point_id, m.summary, m.created_at::date "
        "FROM memories m "
        "JOIN memory_entities me ON me.memory_id = m.id "
        "WHERE me.entity_name ILIKE %s "
        "  AND m.is_deprecated = FALSE "
        "  AND m.source != 'compaction' "
        "ORDER BY m.created_at DESC "
        "LIMIT %s;",
        (topic, limit)
    )

    if not rows:
        return []

    memories = []
    for line in rows.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            memories.append({
                "id": parts[0].strip(),
                "summary": parts[1].strip(),
                "date": parts[2].strip(),
            })
    return memories


def summarize_cluster(topic: str, memories: list) -> str:
    """
    Use LLM to summarize a cluster of memories into a compact digest.
    Uses cheap Haiku model.
    """
    import urllib.request

    # Get API key
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        try:
            key_file = os.path.expanduser("~/.clyde-memory/openrouter.env")
            with open(key_file) as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY="):
                        key = line.strip().split("=", 1)[1]
        except Exception:
            pass
    if not key:
        return ""

    # Build memory list for prompt
    mem_lines = []
    for m in memories[:20]:  # Cap at 20 for token budget
        mem_lines.append(f"- [{m['date']}] {m['summary']}")
    memory_block = "\n".join(mem_lines)

    prompt = f"""Summarize these related memories about "{topic}" into a single concise paragraph.
Preserve: key technical facts, configuration values, decisions made, current state.
Drop: redundant info, debugging steps, timestamps unless they mark a milestone.
Keep it under 200 words.

Memories:
{memory_block}

Summary:"""

    payload = json.dumps({
        "model": "anthropic/claude-haiku-4.5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.1,
    })

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload.encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openclaw/ClydeMemory",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log(f"LLM summarization failed: {e}")
        return ""


def archive_memories(memory_ids: list, reason: str = "compaction"):
    """Mark old memories as deprecated (soft archive)."""
    for mid in memory_ids:
        db.pg_execute(
            "UPDATE memories "
            "SET is_deprecated = TRUE, "
            "    deprecated_reason = %s, "
            "    updated_at = NOW() "
            "WHERE qdrant_point_id = %s "
            "  AND is_deprecated = FALSE;",
            (reason, mid)
        )


def store_digest(topic: str, summary: str):
    """Store the compacted digest as a new memory via Mem0."""
    digest_text = f"[Compacted digest — {topic}] {summary}"
    try:
        # Try daemon first
        import socket as sock
        s = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
        s.settimeout(30)
        s.connect("/tmp/openclaw-memo.sock")
        s.sendall(json.dumps({
            "method": "add",
            "params": {"text": digest_text, "user_id": os.environ.get("CLYDE_USER", "default"), "impact": "high", "writer": "compaction"}
        }).encode() + b"\n")
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        s.close()
        resp = json.loads(data)
        if resp.get("ok"):
            # Mark as compaction source
            if resp.get("result", {}).get("results"):
                new_id = resp["result"]["results"][0].get("id", "")
                if new_id:
                    db.pg_execute("UPDATE memories SET source = 'compaction' WHERE qdrant_point_id = %s;", (new_id,))
            return True
    except Exception:
        pass

    # Fallback: direct Mem0
    try:
        from memo_daemon import _memory
        result = _memory.add(digest_text, user_id=os.environ.get("CLYDE_USER", "default"))
        return True
    except Exception:
        return False


def compact_topic(topic: str, dry_run: bool = False) -> dict:
    """
    Full compaction pipeline for a topic:
    1. Get all raw memories
    2. Summarize via LLM
    3. Store digest
    4. Archive old memories
    5. Set cooldown
    """
    memories = get_memories_for_topic(topic)
    if len(memories) < DENSITY_THRESHOLD:
        return {"status": "below_threshold", "count": len(memories)}

    _log(f"Compacting '{topic}': {len(memories)} memories")

    # Keep the 3 newest memories unarchived (they're still fresh)
    keep_recent = 3
    to_archive = memories[keep_recent:]
    to_summarize = memories  # Summarize all for context

    if dry_run:
        return {
            "status": "dry_run",
            "topic": topic,
            "total": len(memories),
            "would_archive": len(to_archive),
            "would_keep": keep_recent,
            "sample_summaries": [m["summary"][:80] for m in to_summarize[:5]],
        }

    # Summarize
    summary = summarize_cluster(topic, to_summarize)
    if not summary:
        return {"status": "summarization_failed", "topic": topic}

    # Store digest
    stored = store_digest(topic, summary)
    if not stored:
        return {"status": "store_failed", "topic": topic}

    # Archive old memories
    archive_ids = [m["id"] for m in to_archive]
    archive_memories(archive_ids)

    # Set cooldown
    set_cooldown(topic)

    # Log job
    details = json.dumps({"topic": topic, "archived": len(archive_ids), "digest_length": len(summary)})
    db.pg_execute(
        "INSERT INTO agent_jobs (agent_name, job_type, status, finished_at, memories_affected, details) "
        "VALUES ('compaction_trigger', 'topic_compaction', 'completed', NOW(), %s, %s::jsonb);",
        (len(archive_ids), details)
    )

    return {
        "status": "completed",
        "topic": topic,
        "archived": len(archive_ids),
        "kept_recent": keep_recent,
        "digest_length": len(summary),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main run
# ═══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False) -> list:
    """Scan for dense topics and compact them."""
    dense = get_dense_topics()
    if not dense:
        _log("No topics above density threshold")
        return []

    _log(f"Found {len(dense)} dense topics")
    results = []
    compacted = 0

    for topic_info in dense:
        if compacted >= MAX_COMPACTIONS_PER_RUN:
            _log(f"Hit max compactions per run ({MAX_COMPACTIONS_PER_RUN})")
            break

        topic = topic_info["topic"]

        if check_cooldown(topic):
            _log(f"  '{topic}' in cooldown, skipping")
            continue

        result = compact_topic(topic, dry_run=dry_run)
        results.append(result)

        if result["status"] == "completed":
            compacted += 1

    _log(f"Run complete: {compacted} topics compacted")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Density-based memory compaction")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compacted")
    parser.add_argument("--topic", help="Compact a specific topic")
    parser.add_argument("--scan", action="store_true", help="Just scan for dense topics")
    args = parser.parse_args()

    if args.scan:
        dense = get_dense_topics()
        if dense:
            print(f"{'Topic':<30} {'Memories':>8}")
            print("-" * 40)
            for t in dense:
                cooldown = "⏸" if check_cooldown(t["topic"]) else "✓"
                print(f"{t['topic']:<30} {t['count']:>8}  {cooldown}")
        else:
            print("No topics above density threshold")

    elif args.topic:
        result = compact_topic(args.topic, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))

    else:
        results = run(dry_run=args.dry_run)
        for r in results:
            print(json.dumps(r, indent=2))
