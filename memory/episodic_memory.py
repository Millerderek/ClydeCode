#!/usr/bin/env python3
"""
episodic_memory.py — Phase 7B: Episodic Memory + Causal Chains

Preserves coherent multi-turn work sessions as episodes and extracts
causal chains (trigger → investigation → resolution → lesson).

Key differences from existing memory layers:
- Memories (Mem0/Qdrant): individual facts, preferences, decisions
- LCM: raw conversation turns + hierarchical summaries
- Episodes: coherent narrative units spanning one or more sessions,
  capturing WHAT HAPPENED as a story, not fragments

Schema:
  episodes: id, title, summary, outcome, entities[], topics[],
            session_ids[], started_at, ended_at, turn_count, status
  causal_chains: id, episode_id, trigger, investigation, resolution,
                 lesson, confidence, entities[]

Pipeline:
  1. Scan LCM sessions for completed/large sessions not yet episodified
  2. Pull summary nodes (hierarchical DAG) for each session
  3. LLM pass: synthesize summaries into episode narrative + causal chain
  4. Store episode + chain, link to entities
  5. Wire into daemon search path

Cron: episodic_memory.py ingest   (every 4h — process new sessions)
      episodic_memory.py stats    (show episode stats)

Usage:
  from episodic_memory import get_episode_context
"""

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
from config import OPENROUTER_API_KEY, OPENROUTER_KEY_FILE

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

EPISODE_MODEL = "anthropic/claude-haiku-4-5"
MAX_EPISODE_TOKENS = 1200

# Minimum session size to consider for episodification
MIN_TURNS_FOR_EPISODE = 8

# Max summary text to feed to LLM per session
MAX_SUMMARY_CHARS = 4000

# How many recent episodes to surface in context
EPISODE_CONTEXT_LIMIT = 3


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [episodic] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════════

def migrate():
    """Create episode tables. Idempotent."""
    conn = db.get_pg()
    if not conn:
        _log("ERROR: No PG connection")
        return False

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            outcome TEXT DEFAULT '',
            entities TEXT[] DEFAULT '{}',
            topics TEXT[] DEFAULT '{}',
            session_ids TEXT[] DEFAULT '{}',
            started_at TIMESTAMPTZ,
            ended_at TIMESTAMPTZ,
            turn_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'completed',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS causal_chains (
            id SERIAL PRIMARY KEY,
            episode_id INTEGER REFERENCES episodes(id) ON DELETE CASCADE,
            trigger TEXT NOT NULL,
            investigation TEXT DEFAULT '',
            resolution TEXT DEFAULT '',
            lesson TEXT DEFAULT '',
            confidence REAL DEFAULT 0.5,
            entities TEXT[] DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Track which sessions have been processed
    cur.execute("""
        CREATE TABLE IF NOT EXISTS episode_sessions (
            session_id TEXT PRIMARY KEY,
            episode_id INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
            processed_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ended ON episodes(ended_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_entities ON episodes USING GIN(entities)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_causal_chains_episode ON causal_chains(episode_id)")

    conn.commit()
    _log("Migration complete")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# LLM helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_openrouter_key() -> str:
    if OPENROUTER_API_KEY:
        return OPENROUTER_API_KEY
    if OPENROUTER_KEY_FILE.exists():
        for line in OPENROUTER_KEY_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _llm_call(system: str, user: str) -> str:
    """Call OpenRouter with system+user prompt, return raw text."""
    api_key = _get_openrouter_key()
    if not api_key:
        _log("WARNING: No OpenRouter key")
        return ""

    payload = json.dumps({
        "model": EPISODE_MODEL,
        "max_tokens": MAX_EPISODE_TOKENS,
        "temperature": 0.1,
        "messages": [
            {"role": "user", "content": user}
        ],
        "system": system,
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Millerderek/ClydeMemory",
            "X-Title": "ClydeMemory Episodic",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _log(f"LLM call failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Session scanning
# ═══════════════════════════════════════════════════════════════════════════════

def _get_unprocessed_sessions() -> list[dict]:
    """Find LCM sessions large enough for episodification that haven't been processed."""
    conn = db.get_pg()
    if not conn:
        return []

    cur = conn.cursor()

    cur.execute("""
        SELECT s.session_id, s.last_turn, s.started_at, s.last_archive_at
        FROM lcm_sessions s
        LEFT JOIN episode_sessions es ON es.session_id = s.session_id
        WHERE es.session_id IS NULL
          AND s.last_turn >= %s
        ORDER BY s.started_at DESC
        LIMIT 20
    """, (MIN_TURNS_FOR_EPISODE,))

    sessions = []
    for sid, turns, started, archived in cur.fetchall():
        sessions.append({
            "session_id": sid,
            "turns": turns,
            "started_at": started,
            "last_archive_at": archived,
        })

    return sessions


def _get_session_summaries(session_id: str) -> str:
    """Get hierarchical summary text for a session from lcm_summary_nodes."""
    conn = db.get_pg()
    if not conn:
        return ""

    cur = conn.cursor()

    # Get summaries ordered by depth (highest level first) then turn range
    cur.execute("""
        SELECT depth, turn_start, turn_end, summary
        FROM lcm_summary_nodes
        WHERE session_id = %s
        ORDER BY depth DESC, turn_start ASC
    """, (session_id,))

    rows = cur.fetchall()
    if not rows:
        return ""

    # Build text from summaries, preferring high-level ones
    parts = []
    total_chars = 0
    for depth, t_start, t_end, summary in rows:
        if total_chars > MAX_SUMMARY_CHARS:
            break
        prefix = f"[Turns {t_start}-{t_end}, depth={depth}]"
        parts.append(f"{prefix} {summary}")
        total_chars += len(summary)

    return "\n\n".join(parts)


def _get_session_raw_sample(session_id: str, max_turns: int = 20) -> str:
    """Get a sample of raw messages if no summaries exist."""
    conn = db.get_pg()
    if not conn:
        return ""

    cur = conn.cursor()

    # Get last N turns
    cur.execute("""
        SELECT turn_index, role, substr(content, 1, 300)
        FROM lcm_messages
        WHERE session_id = %s
        ORDER BY turn_index DESC
        LIMIT %s
    """, (session_id, max_turns))

    rows = cur.fetchall()
    if not rows:
        return ""

    rows.reverse()  # chronological order
    parts = []
    for turn, role, content in rows:
        parts.append(f"[{role}] {content}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Episode synthesis
# ═══════════════════════════════════════════════════════════════════════════════

EPISODE_SYSTEM_PROMPT = """You are a JSON extraction engine. You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no headers, no commentary.

Analyze this work session between Derek (IT engineer) and Clyde (AI assistant). Output this exact JSON structure:

{"title":"...","summary":"...","outcome":"completed|partial|abandoned|ongoing|diagnostic","entities":["..."],"topics":["..."],"has_causal_chain":true,"causal_chain":{"trigger":"...","investigation":"...","resolution":"...","lesson":"...","confidence":0.8}}

Rules:
- title: max 80 chars, descriptive
- summary: 2-4 sentences, narrative style, what happened and why
- entities: key systems/tools/services/files, max 8
- topics: max 3 from: solar, memory_system, telegram_bot, home_automation, shopping, networking, inverter, justicebot, infrastructure, coding, deployment
- causal_chain: set has_causal_chain=false if session was just Q&A with no problem->solution arc
- lesson: empty string if nothing notable learned

RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT."""


def synthesize_episode(session_id: str, summary_text: str,
                       session_info: dict) -> dict | None:
    """
    Synthesize an episode from session summaries via LLM.

    Returns dict with episode data or None on failure.
    """
    if not summary_text or len(summary_text) < 50:
        return None

    user_prompt = f"""<task>Extract structured data from these session notes into JSON.</task>

<session_notes>
{summary_text[:MAX_SUMMARY_CHARS]}
</session_notes>

<instructions>
Respond with ONLY a JSON object. No markdown. No explanation. Just the JSON.
The JSON must have these keys: title, summary, outcome, entities, topics, has_causal_chain, causal_chain.
Example format: {{"title":"Fixed Docker deploy","summary":"Derek needed to fix...","outcome":"completed","entities":["Docker","nginx"],"topics":["deployment"],"has_causal_chain":true,"causal_chain":{{"trigger":"Deploy failed","investigation":"Checked logs","resolution":"Fixed port binding","lesson":"Always check port conflicts","confidence":0.9}}}}
</instructions>

Respond with ONLY the JSON object:"""

    raw = _llm_call(EPISODE_SYSTEM_PROMPT, user_prompt)
    if not raw:
        return None

    try:
        # Strip markdown fences if present
        clean = re.sub(r'^```json?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
        # Try to find JSON object in the response
        if not clean.startswith('{'):
            # Look for first { and last }
            start = clean.find('{')
            end = clean.rfind('}')
            if start >= 0 and end > start:
                clean = clean[start:end + 1]
            else:
                _log(f"No JSON found for session {session_id[:8]}: {raw[:200]}")
                return None
        result = json.loads(clean)
    except json.JSONDecodeError:
        _log(f"JSON parse failed for session {session_id[:8]}: {raw[:200]}")
        return None

    # Validate required fields
    if not result.get("title") or not result.get("summary"):
        _log(f"Missing required fields for session {session_id[:8]}")
        return None

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════════════════

def store_episode(episode_data: dict, session_id: str,
                  session_info: dict) -> int | None:
    """Store episode + causal chain to PG. Returns episode ID."""
    conn = db.get_pg()
    if not conn:
        return None

    cur = conn.cursor()

    entities = episode_data.get("entities", [])
    topics = episode_data.get("topics", [])

    cur.execute("""
        INSERT INTO episodes
            (title, summary, outcome, entities, topics, session_ids,
             started_at, ended_at, turn_count, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        episode_data["title"],
        episode_data["summary"],
        episode_data.get("outcome", "completed"),
        entities,
        topics,
        [session_id],
        session_info.get("started_at"),
        session_info.get("last_archive_at") or session_info.get("started_at"),
        session_info.get("turns", 0),
        "completed",
    ))

    episode_id = cur.fetchone()[0]

    # Store causal chain if present
    chain = episode_data.get("causal_chain", {})
    if episode_data.get("has_causal_chain", False) and chain.get("trigger"):
        cur.execute("""
            INSERT INTO causal_chains
                (episode_id, trigger, investigation, resolution, lesson,
                 confidence, entities)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            episode_id,
            chain.get("trigger", ""),
            chain.get("investigation", ""),
            chain.get("resolution", ""),
            chain.get("lesson", ""),
            chain.get("confidence", 0.5),
            entities,
        ))

    # Mark session as processed
    cur.execute("""
        INSERT INTO episode_sessions (session_id, episode_id)
        VALUES (%s, %s)
        ON CONFLICT (session_id) DO UPDATE SET
            episode_id = EXCLUDED.episode_id,
            processed_at = NOW()
    """, (session_id, episode_id))

    conn.commit()
    _log(f"  Stored episode #{episode_id}: {episode_data['title'][:60]}")
    return episode_id


# ═══════════════════════════════════════════════════════════════════════════════
# Ingest pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_episodes(max_sessions: int = 10, dry_run: bool = False) -> int:
    """
    Main ingest: find unprocessed sessions → synthesize episodes → store.
    Returns count of episodes created.
    """
    sessions = _get_unprocessed_sessions()
    if not sessions:
        _log("No unprocessed sessions found")
        return 0

    _log(f"Found {len(sessions)} unprocessed sessions, processing up to {max_sessions}")
    created = 0

    for session in sessions[:max_sessions]:
        sid = session["session_id"]
        short_id = sid[:8]

        # Get summaries first, fall back to raw sample
        text = _get_session_summaries(sid)
        if not text:
            text = _get_session_raw_sample(sid)

        if not text or len(text) < 50:
            _log(f"  {short_id}: too short ({len(text) if text else 0} chars), marking skipped")
            if not dry_run:
                # Mark as processed to avoid retrying
                conn = db.get_pg()
                if conn:
                    conn.cursor().execute(
                        "INSERT INTO episode_sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
                        (sid,)
                    )
                    conn.commit()
            continue

        _log(f"  {short_id}: {session['turns']} turns, {len(text)} chars of summary")

        if dry_run:
            print(f"  [dry-run] Would synthesize episode for {short_id}")
            created += 1
            continue

        # Synthesize via LLM
        episode_data = synthesize_episode(sid, text, session)
        if not episode_data:
            _log(f"  {short_id}: synthesis failed, marking processed")
            conn = db.get_pg()
            if conn:
                conn.cursor().execute(
                    "INSERT INTO episode_sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (sid,)
                )
                conn.commit()
            continue

        # Store
        episode_id = store_episode(episode_data, sid, session)
        if episode_id:
            created += 1

            # Link episode entities to the graph
            try:
                from graph_extractor import _upsert_entity
                for ent in episode_data.get("entities", []):
                    _upsert_entity(ent, "general")
            except Exception:
                pass

        # Rate limit LLM calls
        time.sleep(1)

    _log(f"Created {created} episodes from {min(len(sessions), max_sessions)} sessions")
    return created


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def search_episodes(query: str, limit: int = 5) -> list[dict]:
    """
    Search episodes by entity overlap and text matching.
    Returns list of episode dicts with causal chains attached.
    """
    conn = db.get_pg()
    if not conn:
        return []

    cur = conn.cursor()

    # Extract query words for FTS
    words = re.findall(r'[a-zA-Z]{3,}', query)
    if not words:
        return []

    ts_query = " | ".join(words[:8])

    # Search by FTS on title + summary, boosted by entity overlap
    cur.execute("""
        SELECT e.id, e.title, e.summary, e.outcome, e.entities, e.topics,
               e.started_at, e.ended_at, e.turn_count,
               ts_rank(
                   to_tsvector('english', e.title || ' ' || e.summary),
                   plainto_tsquery('english', %s)
               ) as rank
        FROM episodes e
        WHERE to_tsvector('english', e.title || ' ' || e.summary)
              @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC, e.ended_at DESC
        LIMIT %s
    """, (ts_query, ts_query, limit))

    results = []
    for row in cur.fetchall():
        ep = {
            "id": row[0],
            "title": row[1],
            "summary": row[2],
            "outcome": row[3],
            "entities": row[4] or [],
            "topics": row[5] or [],
            "started_at": row[6],
            "ended_at": row[7],
            "turn_count": row[8],
            "rank": round(row[9], 4),
        }

        # Attach causal chain if exists
        cur.execute("""
            SELECT trigger, investigation, resolution, lesson, confidence
            FROM causal_chains
            WHERE episode_id = %s
            LIMIT 1
        """, (ep["id"],))
        chain = cur.fetchone()
        if chain:
            ep["causal_chain"] = {
                "trigger": chain[0],
                "investigation": chain[1],
                "resolution": chain[2],
                "lesson": chain[3],
                "confidence": chain[4],
            }

        results.append(ep)

    return results


def search_episodes_by_entity(entity_names: list[str], limit: int = 5) -> list[dict]:
    """Search episodes that involve specific entities."""
    conn = db.get_pg()
    if not conn:
        return []

    cur = conn.cursor()

    # Use array overlap operator &&
    cur.execute("""
        SELECT e.id, e.title, e.summary, e.outcome, e.entities, e.topics,
               e.started_at, e.ended_at, e.turn_count
        FROM episodes e
        WHERE e.entities && %s
        ORDER BY e.ended_at DESC
        LIMIT %s
    """, (entity_names, limit))

    results = []
    for row in cur.fetchall():
        ep = {
            "id": row[0],
            "title": row[1],
            "summary": row[2],
            "outcome": row[3],
            "entities": row[4] or [],
            "topics": row[5] or [],
            "started_at": row[6],
            "ended_at": row[7],
            "turn_count": row[8],
        }

        cur.execute("""
            SELECT trigger, investigation, resolution, lesson, confidence
            FROM causal_chains WHERE episode_id = %s LIMIT 1
        """, (ep["id"],))
        chain = cur.fetchone()
        if chain:
            ep["causal_chain"] = {
                "trigger": chain[0],
                "investigation": chain[1],
                "resolution": chain[2],
                "lesson": chain[3],
                "confidence": chain[4],
            }

        results.append(ep)

    return results


def get_recent_lessons(limit: int = 5) -> list[dict]:
    """Get recent causal chains with non-empty lessons."""
    conn = db.get_pg()
    if not conn:
        return []

    cur = conn.cursor()

    cur.execute("""
        SELECT c.trigger, c.investigation, c.resolution, c.lesson,
               c.confidence, c.entities,
               e.title, e.ended_at
        FROM causal_chains c
        JOIN episodes e ON e.id = c.episode_id
        WHERE c.lesson != '' AND c.lesson IS NOT NULL
        ORDER BY e.ended_at DESC
        LIMIT %s
    """, (limit,))

    return [
        {
            "trigger": r[0],
            "investigation": r[1],
            "resolution": r[2],
            "lesson": r[3],
            "confidence": r[4],
            "entities": r[5] or [],
            "episode_title": r[6],
            "when": r[7],
        }
        for r in cur.fetchall()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Context building (for daemon injection)
# ═══════════════════════════════════════════════════════════════════════════════

def build_episode_context(query: str, entity_names: list[str] = None,
                          limit: int = EPISODE_CONTEXT_LIMIT) -> str:
    """
    Build an <EPISODE_CONTEXT> block for prompt injection.
    Searches by query text AND entity overlap.
    """
    episodes = []

    # Text search
    text_results = search_episodes(query, limit=limit)
    seen_ids = {e["id"] for e in text_results}
    episodes.extend(text_results)

    # Entity search (supplement)
    if entity_names and len(episodes) < limit:
        entity_results = search_episodes_by_entity(entity_names, limit=limit)
        for ep in entity_results:
            if ep["id"] not in seen_ids and len(episodes) < limit:
                episodes.append(ep)
                seen_ids.add(ep["id"])

    if not episodes:
        return ""

    lines = ["<EPISODE_CONTEXT>"]

    for ep in episodes:
        lines.append(f"Episode: {ep['title']}")
        lines.append(f"  {ep['summary']}")
        if ep.get("outcome"):
            lines.append(f"  Outcome: {ep['outcome']}")

        chain = ep.get("causal_chain")
        if chain:
            lines.append(f"  Trigger: {chain['trigger']}")
            if chain.get("investigation"):
                lines.append(f"  Investigated: {chain['investigation']}")
            if chain.get("resolution"):
                lines.append(f"  Resolved: {chain['resolution']}")
            if chain.get("lesson"):
                lines.append(f"  Lesson: {chain['lesson']}")

        lines.append("")

    lines.append("</EPISODE_CONTEXT>")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Daemon handlers
# ═══════════════════════════════════════════════════════════════════════════════

def handle_episode_search(params: dict) -> dict:
    """Search episodes by query text."""
    query = params.get("query", "")
    limit = params.get("limit", 5)
    if not query:
        return {"ok": False, "error": "Missing 'query'"}

    results = search_episodes(query, limit=limit)
    # Serialize datetimes
    for r in results:
        for k in ("started_at", "ended_at", "when"):
            if k in r and r[k]:
                r[k] = str(r[k])
    return {"ok": True, "episodes": results}


def handle_episode_context(params: dict) -> dict:
    """Build episode context for injection."""
    query = params.get("query", "")
    entities = params.get("entities", [])
    if isinstance(entities, str):
        entities = [entities]

    ctx = build_episode_context(query, entity_names=entities or None)
    return {"ok": True, "context": ctx}


def handle_recent_lessons(params: dict) -> dict:
    """Get recent lessons from causal chains."""
    limit = params.get("limit", 5)
    lessons = get_recent_lessons(limit=limit)
    for l in lessons:
        if l.get("when"):
            l["when"] = str(l["when"])
    return {"ok": True, "lessons": lessons}


def handle_episode_stats(params: dict) -> dict:
    """Return episode statistics."""
    conn = db.get_pg()
    if not conn:
        return {"ok": False, "error": "No PG connection"}

    cur = conn.cursor()

    cur.execute("SELECT count(*) FROM episodes")
    total = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM causal_chains")
    chains = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM causal_chains WHERE lesson != '' AND lesson IS NOT NULL")
    lessons = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM episode_sessions")
    processed = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM lcm_sessions WHERE last_turn >= %s", (MIN_TURNS_FOR_EPISODE,))
    eligible = cur.fetchone()[0]

    cur.execute("""
        SELECT outcome, count(*) FROM episodes GROUP BY outcome ORDER BY count DESC
    """)
    outcomes = {r[0]: r[1] for r in cur.fetchall()}

    return {
        "ok": True,
        "episodes": total,
        "causal_chains": chains,
        "lessons_learned": lessons,
        "sessions_processed": processed,
        "sessions_eligible": eligible,
        "outcomes": outcomes,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Stats (CLI)
# ═══════════════════════════════════════════════════════════════════════════════

def show_stats():
    """Print episode statistics."""
    conn = db.get_pg()
    if not conn:
        print("ERROR: No PG connection")
        return

    cur = conn.cursor()

    cur.execute("SELECT count(*) FROM episodes")
    total = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM causal_chains")
    chains = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM causal_chains WHERE lesson != '' AND lesson IS NOT NULL")
    lessons = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM episode_sessions")
    processed = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM lcm_sessions WHERE last_turn >= %s", (MIN_TURNS_FOR_EPISODE,))
    eligible = cur.fetchone()[0]

    print(f"\n  Episodic Memory Stats")
    print(f"  {'='*50}")
    print(f"  Total episodes:       {total}")
    print(f"  Causal chains:        {chains}")
    print(f"  Lessons learned:      {lessons}")
    print(f"  Sessions processed:   {processed}/{eligible} eligible")

    # Outcome breakdown
    cur.execute("SELECT outcome, count(*) FROM episodes GROUP BY outcome ORDER BY count DESC")
    rows = cur.fetchall()
    if rows:
        print(f"\n  Outcomes:")
        for outcome, count in rows:
            print(f"    {outcome}: {count}")

    # Recent episodes
    cur.execute("""
        SELECT e.title, e.outcome, e.turn_count, e.ended_at,
               (SELECT count(*) FROM causal_chains c WHERE c.episode_id = e.id) as chain_count
        FROM episodes e
        ORDER BY e.ended_at DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    if rows:
        print(f"\n  Recent episodes:")
        for title, outcome, turns, ended, chain_ct in rows:
            chain_tag = " [causal]" if chain_ct > 0 else ""
            print(f"    [{outcome}] {title[:60]} ({turns} turns){chain_tag}")

    # Recent lessons
    cur.execute("""
        SELECT c.lesson, e.title
        FROM causal_chains c
        JOIN episodes e ON e.id = c.episode_id
        WHERE c.lesson != '' AND c.lesson IS NOT NULL
        ORDER BY e.ended_at DESC
        LIMIT 5
    """)
    rows = cur.fetchall()
    if rows:
        print(f"\n  Recent lessons:")
        for lesson, title in rows:
            print(f"    [{title[:30]}] {lesson[:80]}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Episodic Memory — Phase 7B")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("migrate", help="Apply schema migrations")
    sub.add_parser("stats", help="Show episode stats")

    p_ingest = sub.add_parser("ingest", help="Process unprocessed sessions into episodes")
    p_ingest.add_argument("--max", type=int, default=10, help="Max sessions to process")
    p_ingest.add_argument("--dry-run", action="store_true", help="Show what would be processed")

    p_search = sub.add_parser("search", help="Search episodes")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=5)

    p_ctx = sub.add_parser("context", help="Build episode context for query")
    p_ctx.add_argument("query", help="Search query")

    p_lessons = sub.add_parser("lessons", help="Show recent lessons")
    p_lessons.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.command == "migrate":
        migrate()

    elif args.command == "stats":
        show_stats()

    elif args.command == "ingest":
        migrate()
        count = ingest_episodes(max_sessions=args.max, dry_run=args.dry_run)
        if args.dry_run:
            print(f"\n[dry-run] Would create ~{count} episodes")

    elif args.command == "search":
        results = search_episodes(args.query, limit=args.limit)
        for ep in results:
            print(f"[{ep['outcome']}] {ep['title']} (rank={ep['rank']})")
            print(f"  {ep['summary'][:120]}")
            if ep.get("causal_chain"):
                print(f"  Lesson: {ep['causal_chain'].get('lesson', 'none')}")
            print()

    elif args.command == "context":
        ctx = build_episode_context(args.query)
        print(ctx if ctx else "No episode context found")

    elif args.command == "lessons":
        lessons = get_recent_lessons(limit=args.limit)
        for l in lessons:
            print(f"[{l['episode_title']}]")
            print(f"  Trigger: {l['trigger']}")
            if l.get("lesson"):
                print(f"  Lesson: {l['lesson']}")
            print()

    else:
        parser.print_help()
