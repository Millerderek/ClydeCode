#!/usr/bin/env python3
"""
context_decay.py — Unused Context Decay Loop

Closes the gap between what the system retrieves, what the model uses,
and what consistently turns out to be dead weight. Maintains a rolling
per-memory utility score that feeds back into retrieval as a soft factor.

Builds on outcome_logger.py's ml_retrievals + ml_responses tables.
Does NOT hard-delete or aggressively demote — uses gradual decay.

Architecture:
    1. outcome_logger already logs retrievals + labels them (useful/partial/irrelevant/harmful)
    2. This module aggregates those labels into per-memory rolling utility scores
    3. Utility scores feed back into salience_engine as a soft multiplier
    4. Guardrails protect constraints, decisions, and safety-critical memories
    5. Promotion detection flags memories that consistently appear in the same bucket

Tables:
    memory_utility       — rolling utility score per memory
    utility_events       — individual utility deltas (audit trail)
    promotion_candidates — memories flagged for type promotion

Scoring model:
    retrieved + used in successful answer:     +2.0
    retrieved + partially used:                +0.5
    retrieved + unused:                        -0.25
    retrieved + unused (low query relevance):  -0.10  (discounted)
    used in corrected/harmful answer:          -1.0
    constraint present and obeyed:             +1.0   (bonus)
    constraint present and ignored:             0.0   (investigate, don't penalize memory)
    repeatedly retrieved + never used:         accumulating (sum of individual -0.25s)

Rolling window: 60 days. Scores outside the window are expired.

Utility factor (fed to salience_engine):
    utility_factor = clamp(0.5, 1.2, 0.85 + (rolling_score / normalization_constant))
    - Heavily decayed memory: 0.5x salience
    - Neutral memory (no data): 1.0x (no effect)
    - Consistently useful memory: 1.2x boost

Guardrails — these memory types have a floor of 0.75 on utility_factor:
    - Memories with bucket = CONSTRAINTS
    - Memories with impact_category = critical or high
    - Memories involved in supersession (deprecated_by is set)
    - Memories that are pinned

Cron: context_decay.py compute   (every 4h — aggregate new labels into utility scores)
      context_decay.py promote   (daily — scan for promotion candidates)
      context_decay.py stats     (show current state)
      context_decay.py inspect <memory_id>  (show full history for one memory)

Can also be imported:
    from context_decay import get_utility_factor, compute_utility_scores
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Rolling window in days — only events within this window count
ROLLING_WINDOW_DAYS = int(os.environ.get("CLYDE_UTILITY_WINDOW", "60"))

# Score deltas per outcome type
DELTA_USEFUL = 2.0        # label >= 0.9
DELTA_PARTIAL = 0.5       # 0.3 < label < 0.9
DELTA_IRRELEVANT = -0.25  # 0.05 < label <= 0.3
DELTA_IRRELEVANT_WEAK = -0.10  # irrelevant but low query relevance (tail hit)
DELTA_HARMFUL = -1.0      # label <= 0.05
DELTA_CONSTRAINT_OBEYED = 1.0  # constraint bucket + useful

# Utility factor mapping
UTILITY_FLOOR = 0.5       # minimum multiplier (heavily decayed)
UTILITY_CEILING = 1.2     # maximum multiplier (consistently useful)
UTILITY_NEUTRAL = 0.85    # baseline offset
UTILITY_NORM = 10.0       # normalization constant (score / this scales the range)

# Guardrail floor — protected memories never go below this factor
GUARDRAIL_FLOOR = 0.75

# Promotion detection thresholds
PROMO_MIN_RETRIEVALS = 5       # must be retrieved at least this many times
PROMO_MIN_USE_RATE = 0.7       # at least 70% of retrievals must be useful/partial
PROMO_CONSISTENT_BUCKET = 0.8  # at least 80% classified in same bucket

# Tail hit detection: if memory's salience rank is in the bottom half, discount penalty
TAIL_HIT_RANK_THRESHOLD = 3    # rank > this is considered a tail hit


def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [context_decay] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Rolling utility score per memory
CREATE TABLE IF NOT EXISTS memory_utility (
    memory_id TEXT PRIMARY KEY,
    rolling_score REAL DEFAULT 0.0,
    utility_factor REAL DEFAULT 1.0,
    total_retrievals INTEGER DEFAULT 0,
    useful_count INTEGER DEFAULT 0,
    partial_count INTEGER DEFAULT 0,
    irrelevant_count INTEGER DEFAULT 0,
    harmful_count INTEGER DEFAULT 0,
    last_useful_at TIMESTAMPTZ,
    last_retrieved_at TIMESTAMPTZ,
    dominant_bucket TEXT,
    dominant_bucket_pct REAL DEFAULT 0.0,
    is_protected BOOLEAN DEFAULT FALSE,
    protection_reason TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Individual utility events (audit trail)
CREATE TABLE IF NOT EXISTS utility_events (
    id SERIAL PRIMARY KEY,
    memory_id TEXT NOT NULL,
    query TEXT,
    session_id TEXT,
    delta REAL NOT NULL,
    reason TEXT,
    bucket TEXT,
    result_rank INTEGER,
    label REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Promotion candidates: memories that should change type/bucket
CREATE TABLE IF NOT EXISTS promotion_candidates (
    id SERIAL PRIMARY KEY,
    memory_id TEXT NOT NULL,
    current_bucket TEXT,
    suggested_bucket TEXT,
    evidence_count INTEGER DEFAULT 0,
    use_rate REAL DEFAULT 0.0,
    bucket_consistency REAL DEFAULT 0.0,
    status TEXT DEFAULT 'pending',
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(memory_id, suggested_bucket)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memory_utility_factor ON memory_utility(utility_factor);
CREATE INDEX IF NOT EXISTS idx_memory_utility_score ON memory_utility(rolling_score);
CREATE INDEX IF NOT EXISTS idx_memory_utility_protected ON memory_utility(is_protected) WHERE is_protected = TRUE;
CREATE INDEX IF NOT EXISTS idx_utility_events_memory ON utility_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_utility_events_created ON utility_events(created_at);
CREATE INDEX IF NOT EXISTS idx_promotion_candidates_status ON promotion_candidates(status) WHERE status = 'pending';
"""

# Add bucket column to ml_retrievals if missing
MIGRATION_SQL = """
ALTER TABLE ml_retrievals ADD COLUMN IF NOT EXISTS packaged_bucket TEXT;
"""


def init_schema():
    """Create utility tables + migrate ml_retrievals. Idempotent."""
    try:
        db.pg_execute_many(SCHEMA_SQL)
        db.pg_execute(MIGRATION_SQL)
        result = db.pg_query("SELECT count(*) FROM memory_utility")
        return result is not None
    except Exception as e:
        _log(f"Schema init error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Guardrail detection
# ═══════════════════════════════════════════════════════════════════════════════

def _check_protected(memory_id: str) -> tuple[bool, str]:
    """
    Check if a memory should be protected from aggressive decay.
    Returns (is_protected, reason).
    """
    # Check pinned
    result = db.pg_query(
        "SELECT pinned, impact_category, is_deprecated, deprecated_by "
        "FROM memories WHERE id::TEXT LIKE %s LIMIT 1",
        (memory_id[:8] + '%',)
    )
    if result:
        parts = result.strip().split("|")
        if len(parts) >= 4:
            pinned = parts[0].strip() == 't'
            impact = parts[1].strip()
            is_deprecated = parts[2].strip() == 't'
            deprecated_by = parts[3].strip()

            if pinned:
                return True, "pinned"
            if impact in ('critical', 'high'):
                return True, f"impact:{impact}"
            if deprecated_by:
                return True, "has_supersession"

    # Check if memory is typically bucketed as CONSTRAINT
    # (We'll know this after compute runs — check dominant_bucket)
    existing = db.pg_query(
        "SELECT dominant_bucket FROM memory_utility WHERE memory_id = %s",
        (memory_id,)
    )
    if existing and existing.strip() == 'CONSTRAINTS':
        return True, "constraint_memory"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Core: Compute utility scores from labeled retrievals
# ═══════════════════════════════════════════════════════════════════════════════

def compute_utility_scores(batch_size: int = 500) -> dict:
    """
    Process newly labeled ml_retrievals that haven't been scored yet.
    Creates utility_events and updates memory_utility rolling scores.

    Returns dict with processing stats.
    """
    # Find labeled retrievals not yet processed into utility_events.
    # We track this by checking if a utility_event exists for the
    # same memory_id + session_id + created_at window.
    # Simpler: use a watermark approach — process labels newer than
    # the most recent utility_event.
    watermark = db.pg_query(
        "SELECT COALESCE(MAX(created_at), '2020-01-01'::TIMESTAMPTZ) FROM utility_events"
    )
    watermark = watermark.strip() if watermark else '2020-01-01'

    result = db.pg_query(
        "SELECT r.memory_id, r.query, r.session_id, r.label, r.label_reason, "
        "r.result_rank, r.packaged_bucket, r.salience_score, r.labeled_at "
        "FROM ml_retrievals r "
        "WHERE r.label IS NOT NULL "
        "AND r.labeled_at > %s::TIMESTAMPTZ "
        "AND r.memory_id IS NOT NULL "
        "AND r.memory_id != '' "
        "ORDER BY r.labeled_at "
        "LIMIT %s",
        (watermark, batch_size)
    )

    if not result:
        return {"processed": 0, "memories_updated": 0}

    processed = 0
    memories_touched = set()

    for line in result.strip().split("\n"):
        parts = line.split("|")
        if len(parts) < 9:
            continue

        memory_id = parts[0].strip()
        query = parts[1].strip()
        session_id = parts[2].strip()
        label = float(parts[3].strip()) if parts[3].strip() else None
        label_reason = parts[4].strip()
        result_rank = int(parts[5].strip()) if parts[5].strip() else 0
        bucket = parts[6].strip() if parts[6].strip() else None
        salience = float(parts[7].strip()) if parts[7].strip() else 0.0
        labeled_at = parts[8].strip()

        if label is None or not memory_id:
            continue

        # Determine delta
        delta, reason = _compute_delta(label, result_rank, bucket)

        # Record event
        db.pg_execute(
            "INSERT INTO utility_events "
            "(memory_id, query, session_id, delta, reason, bucket, result_rank, label) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (memory_id, query[:500], session_id, delta, reason, bucket, result_rank, label)
        )

        memories_touched.add(memory_id)
        processed += 1

    # Recompute rolling scores for all touched memories
    updated = 0
    for mid in memories_touched:
        if _recompute_memory_utility(mid):
            updated += 1

    return {
        "processed": processed,
        "memories_updated": updated,
        "watermark": watermark,
    }


def _compute_delta(label: float, result_rank: int, bucket: str | None) -> tuple[float, str]:
    """
    Compute the utility delta for a single retrieval outcome.
    Returns (delta, reason).
    """
    if label <= 0.05:
        return DELTA_HARMFUL, "harmful"

    if label >= 0.9:
        # Bonus for constraint memories that were obeyed
        if bucket == 'CONSTRAINTS':
            return DELTA_USEFUL + DELTA_CONSTRAINT_OBEYED, "useful+constraint_obeyed"
        return DELTA_USEFUL, "useful"

    if label > 0.3:
        return DELTA_PARTIAL, "partial"

    # Irrelevant (0.05 < label <= 0.3)
    # Discount penalty for tail hits (low-rank results)
    if result_rank and result_rank > TAIL_HIT_RANK_THRESHOLD:
        return DELTA_IRRELEVANT_WEAK, "irrelevant_tail_hit"
    return DELTA_IRRELEVANT, "irrelevant"


def _recompute_memory_utility(memory_id: str) -> bool:
    """
    Recompute rolling utility score for one memory from its events
    within the rolling window. Updates memory_utility row.
    """
    # Get all events within window
    events = db.pg_query(
        "SELECT delta, label, bucket, created_at FROM utility_events "
        "WHERE memory_id = %s "
        "AND created_at > NOW() - INTERVAL '%s days' "
        "ORDER BY created_at",
        (memory_id, ROLLING_WINDOW_DAYS)
    )

    rolling_score = 0.0
    total = 0
    useful = 0
    partial = 0
    irrelevant = 0
    harmful = 0
    last_useful = None
    last_retrieved = None
    bucket_counts: dict[str, int] = {}

    if events:
        for line in events.strip().split("\n"):
            parts = line.split("|")
            if len(parts) < 4:
                continue

            delta = float(parts[0].strip()) if parts[0].strip() else 0.0
            label = float(parts[1].strip()) if parts[1].strip() else 0.0
            bucket = parts[2].strip() if parts[2].strip() else None
            ts = parts[3].strip()

            rolling_score += delta
            total += 1
            last_retrieved = ts

            if label >= 0.9:
                useful += 1
                last_useful = ts
            elif label > 0.3:
                partial += 1
            elif label > 0.05:
                irrelevant += 1
            else:
                harmful += 1

            if bucket:
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    # Dominant bucket
    dominant_bucket = None
    dominant_pct = 0.0
    if bucket_counts and total > 0:
        dominant_bucket = max(bucket_counts, key=bucket_counts.get)
        dominant_pct = round(bucket_counts[dominant_bucket] / total, 3)

    # Compute utility factor
    raw_factor = UTILITY_NEUTRAL + (rolling_score / UTILITY_NORM)
    utility_factor = max(UTILITY_FLOOR, min(UTILITY_CEILING, raw_factor))

    # Evidence gating: dampen factor toward 1.0 when data is thin
    # N < 3:  force neutral (1.0) — not enough signal
    # 3 <= N <= 6: linear ramp from neutral toward computed factor
    # N > 6:  full computed factor
    EVIDENCE_MIN = 3
    EVIDENCE_FULL = 6
    if total < EVIDENCE_MIN:
        utility_factor = 1.0
    elif total < EVIDENCE_FULL:
        ramp = (total - EVIDENCE_MIN) / (EVIDENCE_FULL - EVIDENCE_MIN)
        utility_factor = 1.0 + ramp * (utility_factor - 1.0)

    # Apply guardrail floor
    is_protected, protection_reason = _check_protected(memory_id)
    if is_protected:
        utility_factor = max(GUARDRAIL_FLOOR, utility_factor)

    # Upsert
    db.pg_execute(
        "INSERT INTO memory_utility "
        "(memory_id, rolling_score, utility_factor, total_retrievals, "
        "useful_count, partial_count, irrelevant_count, harmful_count, "
        "last_useful_at, last_retrieved_at, dominant_bucket, dominant_bucket_pct, "
        "is_protected, protection_reason, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()) "
        "ON CONFLICT (memory_id) DO UPDATE SET "
        "rolling_score = EXCLUDED.rolling_score, "
        "utility_factor = EXCLUDED.utility_factor, "
        "total_retrievals = EXCLUDED.total_retrievals, "
        "useful_count = EXCLUDED.useful_count, "
        "partial_count = EXCLUDED.partial_count, "
        "irrelevant_count = EXCLUDED.irrelevant_count, "
        "harmful_count = EXCLUDED.harmful_count, "
        "last_useful_at = EXCLUDED.last_useful_at, "
        "last_retrieved_at = EXCLUDED.last_retrieved_at, "
        "dominant_bucket = EXCLUDED.dominant_bucket, "
        "dominant_bucket_pct = EXCLUDED.dominant_bucket_pct, "
        "is_protected = EXCLUDED.is_protected, "
        "protection_reason = EXCLUDED.protection_reason, "
        "updated_at = NOW()",
        (memory_id, round(rolling_score, 4), round(utility_factor, 4),
         total, useful, partial, irrelevant, harmful,
         last_useful, last_retrieved,
         dominant_bucket, dominant_pct,
         is_protected, protection_reason)
    )

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval integration: get_utility_factor
# ═══════════════════════════════════════════════════════════════════════════════

def get_utility_factor(memory_id: str) -> float:
    """
    Get the utility factor for a memory. Used by salience_engine as a
    soft multiplier: final_score = salience_score * utility_factor

    Returns 1.0 (neutral) if no data exists for this memory.
    """
    result = db.pg_query(
        "SELECT utility_factor FROM memory_utility WHERE memory_id = %s",
        (memory_id,)
    )
    if result and result.strip():
        try:
            return float(result.strip())
        except ValueError:
            pass
    return 1.0  # neutral — no opinion yet


def get_utility_factors_batch(memory_ids: list[str]) -> dict[str, float]:
    """
    Batch fetch utility factors for multiple memories.
    Returns {memory_id: factor}. Missing memories default to 1.0.
    """
    if not memory_ids:
        return {}

    # Build IN clause safely
    placeholders = ','.join(['%s'] * len(memory_ids))
    result = db.pg_query(
        f"SELECT memory_id, utility_factor FROM memory_utility "
        f"WHERE memory_id IN ({placeholders})",
        tuple(memory_ids)
    )

    factors = {mid: 1.0 for mid in memory_ids}  # default neutral
    if result:
        for line in result.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 2:
                mid = parts[0].strip()
                try:
                    factors[mid] = float(parts[1].strip())
                except ValueError:
                    pass

    return factors


# ═══════════════════════════════════════════════════════════════════════════════
# Bucket logging: record which bucket a memory was packaged into
# ═══════════════════════════════════════════════════════════════════════════════

def log_packaged_buckets(results: list, classified: list):
    """
    After prompt_formatter classifies memories into buckets, call this
    to backfill the packaged_bucket column in ml_retrievals.

    Args:
        results: the search results list (has mem_id/id)
        classified: list of ClassifiedMemory from prompt_formatter
    """
    # Build a map from memory text prefix → bucket
    bucket_map = {}
    for cm in classified:
        # Use first 100 chars of text as key (memory texts are unique enough)
        key = cm.text[:100] if hasattr(cm, 'text') else str(cm)[:100]
        bucket_map[key] = cm.bucket if hasattr(cm, 'bucket') else 'FACTS'

    for r in results:
        mem_text = r.get("memory", "")
        mem_id = r.get("id", r.get("mem_id", ""))
        if not mem_id:
            continue

        # Find matching bucket
        bucket = None
        mem_key = mem_text[:100]
        if mem_key in bucket_map:
            bucket = bucket_map[mem_key]

        if bucket:
            try:
                db.pg_execute(
                    "UPDATE ml_retrievals SET packaged_bucket = %s "
                    "WHERE memory_id = %s "
                    "AND packaged_bucket IS NULL "
                    "AND created_at > NOW() - INTERVAL '5 minutes'",
                    (bucket, mem_id)
                )
            except Exception:
                pass  # non-fatal


# ═══════════════════════════════════════════════════════════════════════════════
# Promotion detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_promotions() -> list[dict]:
    """
    Scan memory_utility for memories that deserve type/bucket promotion.
    A memory qualifies if:
      - Retrieved >= PROMO_MIN_RETRIEVALS times
      - Use rate >= PROMO_MIN_USE_RATE
      - Dominant bucket is consistent >= PROMO_CONSISTENT_BUCKET
      - Current inferred type differs from dominant bucket

    Returns list of promotion candidate dicts.
    """
    result = db.pg_query(
        "SELECT memory_id, total_retrievals, useful_count, partial_count, "
        "dominant_bucket, dominant_bucket_pct, rolling_score "
        "FROM memory_utility "
        "WHERE total_retrievals >= %s "
        "AND dominant_bucket IS NOT NULL "
        "AND dominant_bucket_pct >= %s "
        "ORDER BY rolling_score DESC",
        (PROMO_MIN_RETRIEVALS, PROMO_CONSISTENT_BUCKET)
    )

    if not result:
        return []

    candidates = []
    for line in result.strip().split("\n"):
        parts = line.split("|")
        if len(parts) < 7:
            continue

        memory_id = parts[0].strip()
        total = int(parts[1].strip()) if parts[1].strip() else 0
        useful = int(parts[2].strip()) if parts[2].strip() else 0
        partial = int(parts[3].strip()) if parts[3].strip() else 0
        dominant_bucket = parts[4].strip()
        bucket_pct = float(parts[5].strip()) if parts[5].strip() else 0.0
        score = float(parts[6].strip()) if parts[6].strip() else 0.0

        if total == 0:
            continue

        use_rate = (useful + partial) / total
        if use_rate < PROMO_MIN_USE_RATE:
            continue

        # Check current bucket assignment (from most recent retrieval)
        current = db.pg_query(
            "SELECT packaged_bucket FROM ml_retrievals "
            "WHERE memory_id = %s AND packaged_bucket IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (memory_id,)
        )
        current_bucket = current.strip() if current else 'FACTS'

        # Only flag if the dominant bucket differs from the most common assignment
        # OR if the memory is in FACTS but consistently used as CONSTRAINTS
        if dominant_bucket == 'CONSTRAINTS' and current_bucket != 'CONSTRAINTS':
            suggested = 'CONSTRAINTS'
        elif dominant_bucket == 'PROCEDURES' and current_bucket == 'FACTS':
            suggested = 'PROCEDURES'
        else:
            continue

        # Upsert candidate
        db.pg_execute(
            "INSERT INTO promotion_candidates "
            "(memory_id, current_bucket, suggested_bucket, evidence_count, "
            "use_rate, bucket_consistency) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (memory_id, suggested_bucket) DO UPDATE SET "
            "evidence_count = EXCLUDED.evidence_count, "
            "use_rate = EXCLUDED.use_rate, "
            "bucket_consistency = EXCLUDED.bucket_consistency",
            (memory_id, current_bucket, suggested, total, round(use_rate, 3), bucket_pct)
        )

        candidates.append({
            "memory_id": memory_id,
            "current_bucket": current_bucket,
            "suggested_bucket": suggested,
            "use_rate": round(use_rate, 3),
            "bucket_consistency": bucket_pct,
            "retrievals": total,
            "rolling_score": score,
        })

    _log(f"Promotion scan: {len(candidates)} candidates found")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Window expiry: prune old events outside rolling window
# ═══════════════════════════════════════════════════════════════════════════════

def expire_old_events() -> int:
    """
    Remove utility_events older than the rolling window.
    Returns count of expired events.
    """
    result = db.pg_query(
        "SELECT count(*) FROM utility_events "
        "WHERE created_at < NOW() - INTERVAL '%s days'",
        (ROLLING_WINDOW_DAYS,)
    )
    count = int(result.strip()) if result else 0

    if count > 0:
        db.pg_execute(
            "DELETE FROM utility_events "
            "WHERE created_at < NOW() - INTERVAL '%s days'",
            (ROLLING_WINDOW_DAYS,)
        )
        _log(f"Expired {count} events outside {ROLLING_WINDOW_DAYS}-day window")

    return count


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic: constraint-ignored detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_ignored_constraints() -> list[dict]:
    """
    Find cases where a CONSTRAINT memory was retrieved but went unused.
    These are diagnostic flags — NOT memory quality issues. They may indicate
    a packaging bug or prompt shaping issue.

    Returns list of {memory_id, query, times_ignored, last_ignored}.
    """
    result = db.pg_query(
        "SELECT ue.memory_id, count(*) as ignore_count, max(ue.created_at) as last_at "
        "FROM utility_events ue "
        "WHERE ue.bucket = 'CONSTRAINTS' "
        "AND ue.label <= 0.3 "
        "AND ue.created_at > NOW() - INTERVAL '%s days' "
        "GROUP BY ue.memory_id "
        "HAVING count(*) >= 2 "
        "ORDER BY count(*) DESC "
        "LIMIT 20",
        (ROLLING_WINDOW_DAYS,)
    )

    if not result:
        return []

    flags = []
    for line in result.strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 3:
            flags.append({
                "memory_id": parts[0].strip(),
                "times_ignored": int(parts[1].strip()) if parts[1].strip() else 0,
                "last_ignored": parts[2].strip(),
            })

    if flags:
        _log(f"WARNING: {len(flags)} constraint memories being ignored by model")

    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════════

def get_stats() -> dict:
    """Return decay system statistics."""
    stats = {}

    # Total tracked memories
    r = db.pg_query("SELECT count(*) FROM memory_utility")
    stats["tracked_memories"] = int(r.strip()) if r else 0

    # Distribution by factor range
    dist = db.pg_query(
        "SELECT "
        "count(*) FILTER (WHERE utility_factor >= 1.1), "
        "count(*) FILTER (WHERE utility_factor >= 0.9 AND utility_factor < 1.1), "
        "count(*) FILTER (WHERE utility_factor >= 0.7 AND utility_factor < 0.9), "
        "count(*) FILTER (WHERE utility_factor < 0.7) "
        "FROM memory_utility"
    )
    if dist and "|" in dist:
        parts = dist.split("|")
        stats["factor_distribution"] = {
            "boosted (>=1.1)": int(parts[0].strip() or 0),
            "neutral (0.9-1.1)": int(parts[1].strip() or 0),
            "decayed (0.7-0.9)": int(parts[2].strip() or 0),
            "heavy_decay (<0.7)": int(parts[3].strip() or 0),
        }

    # Protected memories
    r = db.pg_query("SELECT count(*) FROM memory_utility WHERE is_protected = TRUE")
    stats["protected_memories"] = int(r.strip()) if r else 0

    # Total events
    r = db.pg_query("SELECT count(*) FROM utility_events")
    stats["total_events"] = int(r.strip()) if r else 0

    # Events in window
    r = db.pg_query(
        "SELECT count(*) FROM utility_events "
        "WHERE created_at > NOW() - INTERVAL '%s days'",
        (ROLLING_WINDOW_DAYS,)
    )
    stats["events_in_window"] = int(r.strip()) if r else 0

    # Promotion candidates
    r = db.pg_query("SELECT count(*) FROM promotion_candidates WHERE status = 'pending'")
    stats["pending_promotions"] = int(r.strip()) if r else 0

    # Top 5 most useful memories
    top = db.pg_query(
        "SELECT memory_id, rolling_score, utility_factor, useful_count, total_retrievals "
        "FROM memory_utility "
        "ORDER BY rolling_score DESC "
        "LIMIT 5"
    )
    stats["top_memories"] = []
    if top:
        for line in top.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 5:
                stats["top_memories"].append({
                    "id": parts[0].strip()[:12],
                    "score": float(parts[1].strip()) if parts[1].strip() else 0.0,
                    "factor": float(parts[2].strip()) if parts[2].strip() else 1.0,
                    "useful": int(parts[3].strip()) if parts[3].strip() else 0,
                    "total": int(parts[4].strip()) if parts[4].strip() else 0,
                })

    # Bottom 5 most decayed
    bottom = db.pg_query(
        "SELECT memory_id, rolling_score, utility_factor, irrelevant_count, total_retrievals "
        "FROM memory_utility "
        "WHERE total_retrievals > 0 "
        "ORDER BY rolling_score ASC "
        "LIMIT 5"
    )
    stats["bottom_memories"] = []
    if bottom:
        for line in bottom.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 5:
                stats["bottom_memories"].append({
                    "id": parts[0].strip()[:12],
                    "score": float(parts[1].strip()) if parts[1].strip() else 0.0,
                    "factor": float(parts[2].strip()) if parts[2].strip() else 1.0,
                    "irrelevant": int(parts[3].strip()) if parts[3].strip() else 0,
                    "total": int(parts[4].strip()) if parts[4].strip() else 0,
                })

    return stats


def inspect_memory(memory_id: str) -> dict:
    """
    Full diagnostic for a single memory: utility score, all events,
    protection status, promotion candidacy.
    """
    info = {"memory_id": memory_id}

    # Utility record
    r = db.pg_query(
        "SELECT rolling_score, utility_factor, total_retrievals, "
        "useful_count, partial_count, irrelevant_count, harmful_count, "
        "dominant_bucket, dominant_bucket_pct, is_protected, protection_reason, "
        "last_useful_at, last_retrieved_at "
        "FROM memory_utility WHERE memory_id = %s",
        (memory_id,)
    )
    if r:
        parts = r.strip().split("|")
        if len(parts) >= 13:
            info["utility"] = {
                "rolling_score": float(parts[0].strip()) if parts[0].strip() else 0.0,
                "factor": float(parts[1].strip()) if parts[1].strip() else 1.0,
                "total_retrievals": int(parts[2].strip()) if parts[2].strip() else 0,
                "useful": int(parts[3].strip()) if parts[3].strip() else 0,
                "partial": int(parts[4].strip()) if parts[4].strip() else 0,
                "irrelevant": int(parts[5].strip()) if parts[5].strip() else 0,
                "harmful": int(parts[6].strip()) if parts[6].strip() else 0,
                "dominant_bucket": parts[7].strip(),
                "bucket_pct": float(parts[8].strip()) if parts[8].strip() else 0.0,
                "protected": parts[9].strip() == 't',
                "protection_reason": parts[10].strip(),
                "last_useful": parts[11].strip(),
                "last_retrieved": parts[12].strip(),
            }
    else:
        info["utility"] = None

    # Recent events
    events = db.pg_query(
        "SELECT delta, reason, bucket, label, query, created_at "
        "FROM utility_events "
        "WHERE memory_id = %s "
        "ORDER BY created_at DESC "
        "LIMIT 20",
        (memory_id,)
    )
    info["recent_events"] = []
    if events:
        for line in events.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 6:
                info["recent_events"].append({
                    "delta": float(parts[0].strip()) if parts[0].strip() else 0.0,
                    "reason": parts[1].strip(),
                    "bucket": parts[2].strip(),
                    "label": float(parts[3].strip()) if parts[3].strip() else None,
                    "query": parts[4].strip()[:80],
                    "at": parts[5].strip()[:19],
                })

    # Promotion status
    promo = db.pg_query(
        "SELECT current_bucket, suggested_bucket, evidence_count, use_rate, status "
        "FROM promotion_candidates WHERE memory_id = %s",
        (memory_id,)
    )
    if promo:
        parts = promo.strip().split("|")
        if len(parts) >= 5:
            info["promotion"] = {
                "current": parts[0].strip(),
                "suggested": parts[1].strip(),
                "evidence": int(parts[2].strip()) if parts[2].strip() else 0,
                "use_rate": float(parts[3].strip()) if parts[3].strip() else 0.0,
                "status": parts[4].strip(),
            }

    # Memory text (for context)
    mem = db.pg_query(
        "SELECT summary FROM memories WHERE id::TEXT LIKE %s LIMIT 1",
        (memory_id[:8] + '%',)
    )
    if mem:
        info["memory_text"] = mem.strip()[:200]

    return info


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
    print(f"\n  {BOLD}Context Decay — Stats{RESET}\n")
    print(f"  Tracked memories:  {stats.get('tracked_memories', 0)}")
    print(f"  Protected:         {stats.get('protected_memories', 0)}")
    print(f"  Total events:      {stats.get('total_events', 0)}")
    print(f"  Events in window:  {stats.get('events_in_window', 0)} ({ROLLING_WINDOW_DAYS}d)")
    print(f"  Pending promotions:{stats.get('pending_promotions', 0)}")

    dist = stats.get("factor_distribution", {})
    if dist:
        print(f"\n  {BOLD}Utility Factor Distribution:{RESET}")
        print(f"    {GREEN}Boosted (≥1.1):  {dist.get('boosted (>=1.1)', 0)}{RESET}")
        print(f"    Neutral (0.9-1.1): {dist.get('neutral (0.9-1.1)', 0)}")
        print(f"    {YELLOW}Decayed (0.7-0.9): {dist.get('decayed (0.7-0.9)', 0)}{RESET}")
        print(f"    {RED}Heavy decay (<0.7): {dist.get('heavy_decay (<0.7)', 0)}{RESET}")

    top = stats.get("top_memories", [])
    if top:
        print(f"\n  {BOLD}Top 5 Most Useful:{RESET}")
        for m in top:
            print(f"    {GREEN}{m['id']}{RESET}  score={m['score']:+.1f}  "
                  f"factor={m['factor']:.2f}  useful={m['useful']}/{m['total']}")

    bottom = stats.get("bottom_memories", [])
    if bottom:
        print(f"\n  {BOLD}Bottom 5 (Most Decayed):{RESET}")
        for m in bottom:
            print(f"    {RED}{m['id']}{RESET}  score={m['score']:+.1f}  "
                  f"factor={m['factor']:.2f}  irrelevant={m['irrelevant']}/{m['total']}")

    print()


def _cli_compute():
    result = compute_utility_scores()
    print(f"\n  {BOLD}Compute pass complete{RESET}")
    print(f"  Processed:         {result['processed']}")
    print(f"  Memories updated:  {result['memories_updated']}")
    print()


def _cli_promote():
    candidates = detect_promotions()
    if not candidates:
        print(f"  {DIM}No promotion candidates found{RESET}")
        return

    print(f"\n  {BOLD}Promotion Candidates ({len(candidates)}){RESET}\n")
    for c in candidates:
        print(f"  {CYAN}{c['memory_id'][:12]}{RESET}")
        print(f"    {c['current_bucket']} → {GREEN}{c['suggested_bucket']}{RESET}")
        print(f"    use_rate={c['use_rate']:.0%}  consistency={c['bucket_consistency']:.0%}  "
              f"retrievals={c['retrievals']}  score={c['rolling_score']:+.1f}")
        print()


def _cli_inspect(memory_id: str):
    info = inspect_memory(memory_id)

    print(f"\n  {BOLD}Memory: {CYAN}{memory_id}{RESET}\n")

    if info.get("memory_text"):
        print(f"  {DIM}{info['memory_text']}{RESET}\n")

    u = info.get("utility")
    if u:
        protected = f"  {GREEN}PROTECTED ({u['protection_reason']}){RESET}" if u['protected'] else ""
        print(f"  Score:     {u['rolling_score']:+.2f}")
        print(f"  Factor:    {u['factor']:.3f}{protected}")
        print(f"  Retrieved: {u['total_retrievals']}x  "
              f"(useful={u['useful']}, partial={u['partial']}, "
              f"irrelevant={u['irrelevant']}, harmful={u['harmful']})")
        print(f"  Bucket:    {u['dominant_bucket']} ({u['bucket_pct']:.0%})")
        if u['last_useful']:
            print(f"  Last useful: {u['last_useful'][:19]}")
    else:
        print(f"  {DIM}No utility data yet{RESET}")

    promo = info.get("promotion")
    if promo:
        print(f"\n  {BOLD}Promotion:{RESET} {promo['current']} → {GREEN}{promo['suggested']}{RESET}"
              f"  ({promo['status']}, evidence={promo['evidence']}, use_rate={promo['use_rate']:.0%})")

    events = info.get("recent_events", [])
    if events:
        print(f"\n  {BOLD}Recent Events:{RESET}")
        for e in events:
            color = GREEN if e['delta'] > 0 else (RED if e['delta'] < -0.5 else YELLOW)
            print(f"    {e['at']}  {color}{e['delta']:+.2f}{RESET}  {e['reason']}"
                  f"  {DIM}{e['query'][:50]}{RESET}")

    print()


def _cli_constraints():
    flags = detect_ignored_constraints()
    if not flags:
        print(f"  {GREEN}No ignored constraints detected{RESET}")
        return

    print(f"\n  {BOLD}{RED}Ignored Constraints ({len(flags)}){RESET}\n")
    for f in flags:
        print(f"  {CYAN}{f['memory_id'][:12]}{RESET}  "
              f"ignored {f['times_ignored']}x  last: {f['last_ignored'][:19]}")
    print()


def _cli_expire():
    count = expire_old_events()
    print(f"  Expired {count} events outside {ROLLING_WINDOW_DAYS}-day window")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: context_decay.py [init|compute|promote|stats|inspect|constraints|expire]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "init":
        if init_schema():
            print(f"  {GREEN}Schema initialized{RESET}")
        else:
            print(f"  {RED}Schema init failed{RESET}")
    elif cmd == "compute":
        _cli_compute()
    elif cmd == "promote":
        _cli_promote()
    elif cmd == "stats":
        _cli_stats()
    elif cmd == "inspect":
        if len(sys.argv) < 3:
            print("Usage: context_decay.py inspect <memory_id>")
            sys.exit(1)
        _cli_inspect(sys.argv[2])
    elif cmd == "constraints":
        _cli_constraints()
    elif cmd == "expire":
        _cli_expire()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
