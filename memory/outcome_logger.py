#!/usr/bin/env python3
"""
outcome_logger.py -- ML-0: Outcome Logger for OpenClaw.

Collects training data for future ML models by logging every memory retrieval
with its features and scores, then retroactively labeling outcomes based on
whether the memory was actually used in the response.

Architecture:
    1. On search: log_retrieval() stores each candidate with scores + features
    2. On response: log_response() stores the assistant's response text
    3. On next turn: compute_labels() retroactively labels the previous turn's
       retrievals based on semantic overlap with the response

All data lives in PostgreSQL (same DB as the rest of ClydeMemory).
No separate SQLite — we already have PG running.

Usage:
    python3 outcome_logger.py stats          # Show collection stats
    python3 outcome_logger.py unlabeled      # Show unlabeled retrievals
    python3 outcome_logger.py label           # Run labeling pass
    python3 outcome_logger.py export          # Export training data as JSON
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ═══════════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Retrieval log: every memory candidate returned by search
CREATE TABLE IF NOT EXISTS ml_retrievals (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    session_id TEXT,
    turn_number INTEGER DEFAULT 0,
    query TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    memory_text TEXT,
    cosine_score REAL,
    salience_score REAL,
    -- Component scores from salience engine
    score_semantic REAL,
    score_recency REAL,
    score_goal_prox REAL,
    score_oq_boost REAL,
    score_narrative REAL,
    score_working_mode REAL,
    score_frequency REAL,
    score_entity_boost REAL,
    -- Context features
    gate_score REAL,
    working_mode TEXT,
    query_word_count INTEGER,
    query_is_question BOOLEAN DEFAULT FALSE,
    memory_word_count INTEGER,
    result_rank INTEGER,
    -- Outcome (filled retroactively)
    label REAL,                          -- 0.0 = harmful, 0.1 = irrelevant, 0.5 = partial, 1.0 = useful
    label_reason TEXT,                   -- why this label was assigned
    labeled_at TIMESTAMP WITH TIME ZONE,
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Response log: assistant responses for label computation
CREATE TABLE IF NOT EXISTS ml_responses (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    session_id TEXT,
    turn_number INTEGER DEFAULT 0,
    response_text TEXT NOT NULL,
    response_word_count INTEGER,
    has_correction BOOLEAN DEFAULT FALSE,  -- set on next turn
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Training snapshots: model training history
CREATE TABLE IF NOT EXISTS ml_training_log (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    model_name TEXT NOT NULL,
    training_examples INTEGER,
    validation_mae REAL,
    heuristic_mae REAL,
    improvement_pct REAL,
    feature_importance JSONB,
    deployed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_ml_retrievals_session ON ml_retrievals(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_ml_retrievals_unlabeled ON ml_retrievals(label) WHERE label IS NULL;
CREATE INDEX IF NOT EXISTS idx_ml_retrievals_created ON ml_retrievals(created_at);
CREATE INDEX IF NOT EXISTS idx_ml_responses_session ON ml_responses(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_ml_responses_created ON ml_responses(created_at);
"""


def init_schema():
    """Create ML tables if they don't exist."""
    try:
        db.pg_execute_many(SCHEMA_SQL)
        # Verify tables exist
        result = db.pg_query("SELECT count(*) FROM ml_retrievals")
        return result is not None
    except Exception as e:
        print(f"ERROR: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval Logging
# ═══════════════════════════════════════════════════════════════════════════════


def log_retrieval(
    query: str,
    results: list,
    gate_score: float = 1.0,
    session_id: str = None,
    turn_number: int = 0,
):
    """
    Log all retrieval candidates from a search.

    Args:
        query: the search query
        results: list of scored results from salience_engine (or fallback)
        gate_score: context gate score
        session_id: optional session identifier
        turn_number: turn within the session
    """
    if not results:
        return

    query_wc = len(query.split())
    query_is_q = query.rstrip().endswith("?")

    # Detect working mode
    working_mode = "general"
    try:
        from context_gate import classify_topic
        working_mode = classify_topic(query)
    except Exception:
        pass

    # Insert each retrieval row with parameterized queries
    for rank, r in enumerate(results, 1):
        mem_text = r.get("memory", "")
        breakdown = r.get("breakdown", {})
        rid = str(uuid4())
        mid = r.get("id", r.get("mem_id", ""))

        try:
            db.pg_execute(
                "INSERT INTO ml_retrievals ("
                "id, session_id, turn_number, query, memory_id, memory_text, "
                "cosine_score, salience_score, "
                "score_semantic, score_recency, score_goal_prox, score_oq_boost, "
                "score_narrative, score_working_mode, score_frequency, score_entity_boost, "
                "gate_score, working_mode, query_word_count, query_is_question, "
                "memory_word_count, result_rank"
                ") VALUES (%s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s)",
                (rid, session_id, turn_number, query,
                 mid, mem_text[:2000],
                 r.get('score', 0.0), r.get('final', 0.0),
                 breakdown.get('semantic', 0.0), breakdown.get('recency', 0.0),
                 breakdown.get('goal_prox', 0.0), breakdown.get('oq_boost', 0.0),
                 breakdown.get('narrative', 0.0), breakdown.get('working_mode', 0.0),
                 breakdown.get('frequency', 0.0), breakdown.get('entity_boost', 0.0),
                 gate_score, working_mode, query_wc, query_is_q,
                 len(mem_text.split()), rank)
            )
        except Exception:
            pass  # Non-fatal — don't break search over logging


def log_response(
    response_text: str,
    session_id: str = None,
    turn_number: int = 0,
):
    """
    Log an assistant response for retroactive label computation.

    Args:
        response_text: the full assistant response
        session_id: optional session identifier
        turn_number: turn within the session
    """
    if not response_text:
        return

    rid = str(uuid4())
    # Truncate response for storage (keep first 10k chars)
    resp_truncated = response_text[:10000]
    wc = len(response_text.split())

    try:
        db.pg_execute(
            "INSERT INTO ml_responses (id, session_id, turn_number, response_text, response_word_count) "
            "VALUES (%s, %s, %s, %s, %s)",
            (rid, session_id, turn_number, resp_truncated, wc)
        )
    except Exception:
        pass  # Non-fatal


# ═══════════════════════════════════════════════════════════════════════════════
# Label Computation
# ═══════════════════════════════════════════════════════════════════════════════

def _word_overlap_score(text_a: str, text_b: str) -> float:
    """
    Compute word overlap between two texts.
    Returns 0.0-1.0 based on significant word overlap.
    Uses meaningful words (3+ chars, not stopwords).
    """
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "his", "how", "its", "may", "new", "now", "old", "see",
        "way", "who", "did", "get", "let", "say", "she", "too",
        "use", "this", "that", "with", "have", "from", "they",
        "been", "some", "what", "when", "will", "more", "into",
        "also", "than", "them", "very", "just", "about", "which",
        "their", "there", "would", "could", "should", "where",
        "like", "been", "only", "then", "each", "make", "made",
        "does", "done", "most", "such", "here", "much", "many",
    }

    def extract_words(text):
        return set(
            w for w in re.findall(r'[a-z]{3,}', text.lower())
            if w not in stopwords
        )

    words_a = extract_words(text_a)
    words_b = extract_words(text_b)

    if not words_a or not words_b:
        return 0.0

    overlap = words_a & words_b
    # Jaccard-style but weighted toward memory coverage
    # (what fraction of memory words appear in response)
    coverage = len(overlap) / len(words_a) if words_a else 0.0
    return min(1.0, coverage)


def _is_correction(text: str) -> bool:
    """Check if a user message contains a correction signal."""
    correction_patterns = [
        r"\b(no,?\s+(?:it|that|this|actually))",
        r"\b(that'?s (?:wrong|incorrect|not right|not what))",
        r"\b(actually,?\s+(?:it|the|that))",
        r"\bwrong\b",
        r"\b(i meant|i said|i asked for)",
        r"\b(not what i (?:meant|wanted|asked))",
        r"\b(correction|correcting)",
    ]
    for pat in correction_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def compute_labels(batch_size: int = 100) -> dict:
    """
    Retroactively label unlabeled retrievals by matching them against
    the response text from the same session/turn.

    Returns:
        dict with counts: labeled, skipped (no matching response), total_unlabeled
    """
    # Find unlabeled retrievals that have a matching response
    result = db.pg_query(
        "SELECT r.id, r.memory_text, r.session_id, r.turn_number, "
        "resp.response_text, r.cosine_score "
        "FROM ml_retrievals r "
        "JOIN ml_responses resp "
        "ON r.session_id = resp.session_id "
        "AND r.turn_number = resp.turn_number "
        "WHERE r.label IS NULL "
        "AND r.session_id IS NOT NULL "
        "LIMIT %s",
        (batch_size,)
    )

    if not result:
        remaining = db.pg_query("SELECT count(*) FROM ml_retrievals WHERE label IS NULL")
        return {"labeled": 0, "remaining_unlabeled": int(remaining or 0), "batch_size": batch_size}

    labeled = 0
    for line in result.split("\n"):
        parts = line.split("|")
        if len(parts) < 6:
            continue

        rid = parts[0]
        mem_text = parts[1]
        session_id = parts[2]
        turn_number = int(parts[3]) if parts[3] else 0
        response_text = parts[4]
        cosine = float(parts[5]) if parts[5] else 0.0

        if not mem_text or not response_text:
            continue

        # Check for correction in the NEXT turn's user message
        # (responses table may store user follow-ups too, or we check retrievals)
        next_resp = db.pg_query(
            "SELECT response_text FROM ml_responses "
            "WHERE session_id = %s AND turn_number = %s",
            (session_id, turn_number + 1)
        )

        correction_followed = False
        if next_resp:
            correction_followed = _is_correction(next_resp)

        # Compute overlap
        overlap = _word_overlap_score(mem_text, response_text)

        # Assign label
        if correction_followed and overlap > 0.3:
            label = 0.0
            reason = f"harmful: overlap={overlap:.2f}, correction followed"
        elif overlap > 0.5:
            label = 1.0
            reason = f"useful: overlap={overlap:.2f}"
        elif overlap > 0.2:
            label = 0.5
            reason = f"partial: overlap={overlap:.2f}"
        else:
            label = 0.1
            reason = f"irrelevant: overlap={overlap:.2f}"

        db.pg_execute(
            "UPDATE ml_retrievals "
            "SET label = %s, label_reason = %s, labeled_at = NOW() "
            "WHERE id = %s",
            (label, reason, rid)
        )
        labeled += 1

    remaining = db.pg_query("SELECT count(*) FROM ml_retrievals WHERE label IS NULL")

    return {
        "labeled": labeled,
        "remaining_unlabeled": int(remaining or 0),
        "batch_size": batch_size,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Correction rate lookback (CG remaining item)
# ═══════════════════════════════════════════════════════════════════════════════

def get_correction_rate(days=7) -> float:
    """
    Return the fraction of labeled retrievals marked harmful (label=0.0)
    in the last N days. Used by the confidence gate to raise thresholds
    when corrections are frequent.

    Returns 0.0 if no labeled data exists yet.
    """
    total = db.pg_query(
        "SELECT count(*) FROM ml_retrievals "
        "WHERE label IS NOT NULL "
        "AND labeled_at > NOW() - INTERVAL '%s days'",
        (int(days),)
    )
    harmful = db.pg_query(
        "SELECT count(*) FROM ml_retrievals "
        "WHERE label < 0.05 "
        "AND labeled_at > NOW() - INTERVAL '%s days'",
        (int(days),)
    )
    try:
        t = int(total or 0)
        h = int(harmful or 0)
        if t == 0:
            return 0.0
        return round(h / t, 4)
    except (ValueError, ZeroDivisionError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Stats & Export
# ═══════════════════════════════════════════════════════════════════════════════

def get_stats() -> dict:
    """Return collection statistics."""
    stats = {}

    # Total retrievals
    r = db.pg_query("SELECT count(*) FROM ml_retrievals")
    if not r and r != "0":
        return {"error": "no PG connection or table missing"}
    stats["total_retrievals"] = int(r or 0)

    # Labeled vs unlabeled
    stats["labeled"] = int(db.pg_query("SELECT count(*) FROM ml_retrievals WHERE label IS NOT NULL") or 0)
    stats["unlabeled"] = stats["total_retrievals"] - stats["labeled"]

    # Label distribution
    dist = db.pg_query(
        "SELECT "
        "count(*) FILTER (WHERE label > 0.9), "
        "count(*) FILTER (WHERE label > 0.3 AND label < 0.7), "
        "count(*) FILTER (WHERE label > 0.05 AND label < 0.3), "
        "count(*) FILTER (WHERE label < 0.05) "
        "FROM ml_retrievals WHERE label IS NOT NULL"
    )
    if dist and "|" in dist:
        parts = dist.split("|")
        stats["labels"] = {
            "useful": int(parts[0] or 0),
            "partial": int(parts[1] or 0),
            "irrelevant": int(parts[2] or 0),
            "harmful": int(parts[3] or 0),
        }
    else:
        stats["labels"] = {"useful": 0, "partial": 0, "irrelevant": 0, "harmful": 0}

    # Responses
    stats["total_responses"] = int(db.pg_query("SELECT count(*) FROM ml_responses") or 0)

    # Sessions
    stats["sessions"] = int(db.pg_query(
        "SELECT count(DISTINCT session_id) FROM ml_retrievals WHERE session_id IS NOT NULL"
    ) or 0)

    # Date range
    dr = db.pg_query("SELECT min(created_at), max(created_at) FROM ml_retrievals")
    if dr and "|" in dr:
        parts = dr.split("|")
        stats["first_retrieval"] = parts[0][:19] if parts[0] else None
        stats["last_retrieval"] = parts[1][:19] if parts[1] else None

    # Training readiness
    stats["salience_ready"] = stats["labeled"] >= 200
    stats["salience_progress"] = f"{stats['labeled']}/200"

    return stats


def export_training_data(min_label_count: int = 50) -> list:
    """
    Export labeled retrievals as training-ready feature vectors.

    Returns list of dicts, each with:
        - features: dict of numeric features
        - label: float outcome label
    """
    result = db.pg_query(
        "SELECT "
        "cosine_score, salience_score, "
        "score_semantic, score_recency, score_goal_prox, score_oq_boost, "
        "score_narrative, score_working_mode, score_frequency, score_entity_boost, "
        "gate_score, query_word_count, query_is_question, "
        "memory_word_count, result_rank, working_mode, "
        "label "
        "FROM ml_retrievals "
        "WHERE label IS NOT NULL "
        "ORDER BY created_at"
    )

    if not result:
        return []

    rows = result.split("\n")
    if len(rows) < min_label_count:
        return []

    data = []
    for line in rows:
        parts = line.split("|")
        if len(parts) < 17:
            continue

        def _f(idx):
            try:
                return float(parts[idx]) if parts[idx] else 0.0
            except (ValueError, IndexError):
                return 0.0

        data.append({
            "features": {
                "cosine_score": _f(0),
                "salience_score": _f(1),
                "score_semantic": _f(2),
                "score_recency": _f(3),
                "score_goal_prox": _f(4),
                "score_oq_boost": _f(5),
                "score_narrative": _f(6),
                "score_working_mode": _f(7),
                "score_frequency": _f(8),
                "score_entity_boost": _f(9),
                "gate_score": _f(10),
                "query_word_count": int(_f(11)),
                "query_is_question": 1 if parts[12].strip() == "t" else 0,
                "memory_word_count": int(_f(13)),
                "result_rank": int(_f(14)),
                "working_mode": parts[15].strip() or "general",
            },
            "label": _f(16),
        })

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RED = "\033[91m"
RESET = "\033[0m"


def _cli_stats():
    stats = get_stats()
    if "error" in stats:
        print(f"  {RED}{stats['error']}{RESET}")
        return

    print(f"\n  {BOLD}ML Outcome Logger — Stats{RESET}\n")
    print(f"  Retrievals:  {stats['total_retrievals']}")
    print(f"  Labeled:     {stats['labeled']}")
    print(f"  Unlabeled:   {stats['unlabeled']}")
    print(f"  Responses:   {stats['total_responses']}")
    print(f"  Sessions:    {stats['sessions']}")

    if stats.get("first_retrieval"):
        print(f"\n  First:       {stats['first_retrieval']}")
        print(f"  Last:        {stats['last_retrieval']}")

    labels = stats.get("labels", {})
    if any(labels.values()):
        print(f"\n  {BOLD}Label Distribution:{RESET}")
        print(f"    Useful (1.0):     {GREEN}{labels.get('useful', 0)}{RESET}")
        print(f"    Partial (0.5):    {YELLOW}{labels.get('partial', 0)}{RESET}")
        print(f"    Irrelevant (0.1): {DIM}{labels.get('irrelevant', 0)}{RESET}")
        print(f"    Harmful (0.0):    {RED}{labels.get('harmful', 0)}{RESET}")

    print(f"\n  {BOLD}Salience Model Readiness:{RESET}")
    ready = f"{GREEN}READY" if stats["salience_ready"] else f"{YELLOW}COLLECTING"
    print(f"    Progress: {stats['salience_progress']} labeled  [{ready}{RESET}]")
    print()


def _cli_label():
    result = compute_labels()
    if "error" in result:
        print(f"  {RED}{result['error']}{RESET}")
        return
    print(f"\n  {BOLD}Labeling pass complete{RESET}")
    print(f"  Labeled this run:    {result['labeled']}")
    print(f"  Remaining unlabeled: {result['remaining_unlabeled']}")
    print()


def _cli_export():
    data = export_training_data(min_label_count=0)
    if not data:
        print(f"  {YELLOW}No labeled data to export{RESET}")
        return
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: outcome_logger.py [init|stats|label|export]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "init":
        if init_schema():
            print(f"  {GREEN}Schema initialized{RESET}")
        else:
            print(f"  {RED}Schema init failed{RESET}")
    elif cmd == "stats":
        _cli_stats()
    elif cmd == "label":
        _cli_label()
    elif cmd == "export":
        _cli_export()
    elif cmd == "unlabeled":
        result = db.pg_query("SELECT count(*) FROM ml_retrievals WHERE label IS NULL")
        print(f"  Unlabeled retrievals: {result}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
